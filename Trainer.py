import torch
from torch import nn
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from lrschedule import noam_learning_rate_decay
from util import logit, masked_mean, sequence_mask, prepare_spec_image
import audio

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss(size_average=False)

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)
            raise RuntimeError("Mask is None")

        # (B, T, D)
        mask_ = mask.expand_as(input)
        loss = self.criterion(input * mask_, target * mask_)
        return loss / mask_.sum()

class Trainer:
    def __init__(self, model, train_loader, valid_loader, device, args):
        self.model = model
        self.args = args
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learn_rate, betas=(0.5, 0.9),
                                          eps=1e-06, weight_decay=0.0, amsgrad=False)
        self.checkpoint_interval = args.checkpoint_inverval
        self.eval_interval = args.checkpoint_inverval
        self.w = args.binary_divergence_weight
        self.epoch = args.epoch

    def spec_loss(self, y_hat, y, mask, priority_bin=None, priority_w=0):
        masked_l1 = MaskedL1Loss()
        l1 = nn.L1Loss()

        w = self.args.masked_loss_weight

        # L1 loss
        if w > 0:
            assert mask is not None
            l1_loss = w * masked_l1(y_hat, y, mask=mask) + (1 - w) * l1(y_hat, y)
        else:
            assert mask is None
            l1_loss = l1(y_hat, y)

        # Priority L1 loss
        if priority_bin is not None and priority_w > 0:
            if w > 0:
                priority_loss = w * masked_l1(
                    y_hat[:, :, :priority_bin], y[:, :, :priority_bin], mask=mask) \
                                + (1 - w) * l1(y_hat[:, :, :priority_bin], y[:, :, :priority_bin])
            else:
                priority_loss = l1(y_hat[:, :, :priority_bin], y[:, :, :priority_bin])
            l1_loss = (1 - priority_w) * l1_loss + priority_w * priority_loss

        # Binary divergence loss
        if self.w <= 0:
            binary_div = y.data.new(1).zero_()
        else:
            y_hat_logits = logit(y_hat)
            z = -y * y_hat_logits + torch.log1p(torch.exp(y_hat_logits))
            if w > 0:
                binary_div = w * masked_mean(z, mask) + (1 - w) * z.mean()
            else:
                binary_div = z.mean()

        return l1_loss, binary_div


    def train(self, global_step=0, global_epoch=1):
        while global_epoch < self.epoch:
            running_loss = 0.
            for step, (melX, melY, lengths) in enumerate(tqdm(self.train_loader)):
                self.model.train()

                # Learn rate scheduler
                current_lr = noam_learning_rate_decay(self.args.learn_rate, global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                self.optimizer.zero_grad()

                # Transform data to CUDA device
                melX = melX.to(self.device)
                melY = melY.to(self.device)
                lengths = lengths.to(self.device)

                target_mask = sequence_mask(lengths, max_len=melY.size(1)).unsqueeze(-1)

                # Apply model
                melX_output = self.model(melX) # TODO : code model

                # Losses
                mel_l1_loss, mel_binary_div = self.spec_loss(melX_output, melY, target_mask)
                loss = (1 - self.w) * mel_l1_loss + self.w * mel_binary_div

                # Update
                loss.backward()

                # Logs
                self.writer.add_scalar("loss", float(loss.item()), global_step)
                self.writer.add_scalar("mel_l1_loss", float(mel_l1_loss.item()), global_step)
                self.writer.add_scalar("mel_binary_div_loss", float(mel_binary_div.item()), global_step)
                self.writer.add_scalar("learning rate", current_lr, global_step)

                global_step += 1
                running_loss += loss.item()

            if (global_epoch % self.checkpoint_interval == 0):
                self.save_states(global_epoch, melX_output, melX, melY, lengths)
                self.save_checkpoint(global_step, global_epoch)

            self.eval_model(global_epoch)
            avg_loss = running_loss / len(self.train_loader)
            self.writer.add_scalar("train loss (per epoch)", avg_loss, global_epoch)
            print("Train Loss: {}".format(avg_loss))
            global_epoch += 1


    def eval_model(self, global_epoch):
        running_loss = 0.
        for step, (melX, melY, lengths) in enumerate(self.valid_loader):
            self.model.eval()
            melX = melX.to(self.device)
            melY = melY.to(self.device)
            lengths = lengths.to(self.device)
            target_mask = sequence_mask(lengths, max_len=melY.size(1)).unsqueeze(-1)

            melX_outputs = self.model(melX)

            mel_l1_loss, mel_binary_div = self.spec_loss(melX_outputs, melY, target_mask)
            loss = (1 - self.w) * mel_l1_loss + self.w * mel_binary_div

            running_loss += loss.item()

        if global_epoch % self.eval_interval == 0:
            idx = min(1, len(lengths) - 1)
            mel_output = melX_outputs[idx].cpu().data.numpy()
            mel_output = prepare_spec_image(audio._denormalize(mel_output))
            self.writer.add_image("(Eval) Predicted mel spectrogram", mel_output, global_epoch)

            # Target mel spectrogram
            melY = melY[idx].cpu().data.numpy()
            melY = prepare_spec_image(audio._denormalize(melY))
            self.writer.add_image("(Eval) Target mel spectrogram", melY, global_epoch)
            melX = melX[idx].cpu().data.numpy()
            melX = prepare_spec_image(audio._denormalize(melX))
            self.writer.add_image("(Eval) Source mel spectrogram", melX, global_epoch)

        avg_loss = running_loss / len(self.valid_loader)
        self.writer.add_scalar("valid loss (per epoch)", avg_loss, global_epoch)
        print("Valid Loss: {}".format(avg_loss))

    def save_states(self, global_epoch, melX_outputs, melX, melY, lengths):
        print("Save intermediate states at epoch {}".format(global_epoch))

        # idx = np.random.randint(0, len(input_lengths))
        idx = min(1, len(lengths) - 1)

        # Predicted mel spectrogram
        mel_output = melX_outputs[idx].cpu().data.numpy()
        mel_output = prepare_spec_image(audio._denormalize(mel_output))
        self.writer.add_image("Predicted mel spectrogram", mel_output, global_epoch)

        # Target mel spectrogram
        melY = melY[idx].cpu().data.numpy()
        melY = prepare_spec_image(audio._denormalize(melY))
        self.writer.add_image("Target mel spectrogram", melY, global_epoch)
        melX = melX[idx].cpu().data.numpy()
        melX = prepare_spec_image(audio._denormalize(melX))
        self.writer.add_image("Source mel spectrogram", melX, global_epoch)


    def save_checkpoint(self, global_step, global_epoch):
        checkpoint_path = join(self.args.checkpoint_dir, "checkpoint_epoch{:09d}.pth".format(global_epoch))
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": global_step,
            "global_epoch": global_epoch,
        }, checkpoint_path)
        print("Saved checkpoint:", checkpoint_path)






