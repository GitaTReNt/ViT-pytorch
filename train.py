import os
import logging
import random
import numpy as np
import torch.nn as nn
from datetime import timedelta
from argparse import ArgumentParser
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from model import VisionTransformer, CONFIGS
from scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from data_utils import get_loader
from dataclasses import dataclass
import warnings
import numpy as np
import torch.nn.functional as F


# 忽略所有警告
warnings.filterwarnings("ignore")

# 或者仅忽略特定类型的警告，例如 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))



@dataclass
class TrainingArguments:
    learning_rate: float = 3e-2
    weight_decay: float = 0
    num_steps: int = 2000
    decay_type: str = "cosine"
    warmup_steps: int = 400
    max_grad_norm: float = 1.0
    fp16_opt_level: str = 'O0'
    loss_scale: float = 0
    name: str = 'CIFAR'
    model_type: str = "ViT-B_32"
    pretrained_dir: str = "ViT-B_32.npz"
    output_dir: str = "output"
    seed: int = 42
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    img_size: int = 224
    train_batch_size: int = 512
    eval_batch_size: int = 64
    eval_every: int = 100
    num_classes: int = 100


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    config = CONFIGS[args.model_type]
    num_classes = 100
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    eval_losses = AverageMeter()
    correct = 0
    total = 0
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    all_preds, all_label = [], []

    # Initialize the progress bar
    epoch_iterator = tqdm(test_loader, desc="Validating...", bar_format="{l_bar}{bar:10}{r_bar}", dynamic_ncols=True)

    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)

        # Update the progress bar
        epoch_iterator.set_postfix(loss=eval_losses.avg, accuracy=correct / total)

    # Calculate final accuracy
    accuracy = correct / total

    # Print final validation results
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    # Write to TensorBoard
    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)

    return accuracy


def train(args, model):
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_loader, test_loader = get_loader(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=0.9, weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Train batch size: %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    criterion = LabelSmoothingLoss(classes=args.num_classes, smoothing=0.1)
    for _ in range(args.num_steps):
        model.train()
        epoch_iterator = tqdm(train_loader, desc=f"Training ({global_step}/{t_total} Steps)",
                              bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                              dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch

            # Apply Mixup
            x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)

            # Forward pass
            logits, _ = model(x)  # Extract logits from the tuple
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_iterator.set_description(
                    f"Training ({global_step}/{t_total} Steps) - Loss: {losses.val:.4f}"
                )
                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()
                if global_step >= t_total:
                    break
        losses.reset()
        if global_step >= t_total:
            break
    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = ArgumentParser()
    parser.add_argument("--config_file", default="config.yaml", type=str, help="Path to the config file.")
    args = parser.parse_args()
    # Load arguments from the config file
    import yaml
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    args = TrainingArguments(**config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s ,fp16 training: %s" % (args.device, args.fp16))
    set_seed(args)
    model = setup(args)
    train(args, model)


if __name__ == "__main__":
    main()
