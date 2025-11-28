Run the following command in the project directory to make the script run:

>$ git clone https://github.com/huggingface/lerobot.git

Then download and copy model.safetensor and config.json from https://huggingface.co/lerobot/act_aloha_sim_insertion_human/tree/main in old weights

>$ python ./lerobot/src/lerobot/processor/migrate_policy_normalization.py --pretrained-path ./old\ weights/  --output-dir ./migrated_output/
This will migrate old policy to new

To run the script:
>$ python pretrained
