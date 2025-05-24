from pprint import pprint

from capybara import dump_json, get_curdir
from chameleon import calculate_flops
from otter import build_dataset, build_trainer, load_model_from_config

from . import dataset as ds
from . import model as net

DIR = get_curdir(__file__)

def main_train(cfg_name: str):
    model, cfg = load_model_from_config(
        root=DIR, stem='config', cfg_name=cfg_name, network=net)
    train_data, valid_data = build_dataset(cfg, ds)
    trainer = build_trainer(cfg, root=DIR)

    # Add the directory to .gitignore if it exists
    if (f_ign := DIR / '.gitignore').is_file():
        with open(str(f_ign), 'r') as file:
            content = file.read()

        # 檢查是否已包含所需的文本
        entry = f'\n{cfg.name}/\n'
        if entry not in content:
            # 如果没有包含，则將其添加到 .gitignore 文件
            with open(str(f_ign), 'a') as f:
                f.write(f'\n{cfg.name}/\n')

    # -- Log model meta data -- #
    flops, macs, params = calculate_flops(
        model,
        input_shape=(1, 3, *cfg.global_settings.image_size),
        print_detailed=False,
        print_results=False
    )

    meta_data = {
        'FLOPs': flops,
        'MACs': macs,
        'Params': params,
    }

    dump_json(
        meta_data,
        DIR / cfg.name / cfg.name_ind / 'logger' / 'model_meta_data.json'
    )
    pprint(meta_data)
    # ------------------------- #

    restore_all_states = getattr(cfg.common, 'restore_all_states', False)
    model.train()
    trainer.fit(
        model,
        train_dataloaders=train_data,
        val_dataloaders=valid_data,
        ckpt_path=cfg.common.checkpoint_path if restore_all_states else None,
    )
