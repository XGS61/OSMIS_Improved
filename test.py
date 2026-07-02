import config
from core import dataloading, models, utils, tracking


# --- read options --- #
opt = config.read_arguments(train=False)

# --- create dataloader and recommended model config --- #
dataloader, model_config = dataloading.prepare_dataloading(opt)

# --- create models, losses, and optimizers ---#
netG, netD, netEMA = models.create_models(opt, model_config)

# --- create utils --- #
visualizer = tracking.visualizer(opt)

# --- generate images and masks --- #
data_iterator = iter(dataloader)
for i in range(opt.num_generated):
    batch = next(data_iterator)
    batch = utils.preprocess_real(batch, model_config["num_blocks_d0"], opt.device)
    target_mask = batch["masks"][:1]
    target_structure = batch["structures"][:1]
    style_index = i % batch["masks"].shape[0]
    style_image = batch["images"][-1][style_index:style_index + 1]
    style_mask = batch["masks"][style_index:style_index + 1]
    z = utils.sample_noise(opt.noise_dim, 1).to(opt.device)
    fake = (
        netEMA.generate(
            z,
            masks=target_mask,
            structures=target_structure,
            style_images=style_image,
            style_masks=style_mask,
        )
        if not opt.no_EMA
        else netG.generate(
            z,
            masks=target_mask,
            structures=target_structure,
            style_images=style_image,
            style_masks=style_mask,
        )
    )
    visualizer.save_batch(fake, opt.continue_epoch, i=str(i))
