# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
from copy import copy
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_dir", default='exp', type=str)
    parser.add_argument("-train_name")
    parser.add_argument("-train_config")
    parser.add_argument("-data_config")
    parser.add_argument("-num_gpu", default=-1, type=int,
                    metavar='N', help='number of gpu to use, -1 means cpu, 0 means no paraller')
    parser.add_argument("-task_file", default=None, type=str,
                        help="path of task files")
    parser.add_argument("-task_name", default=None, type=str,
                        help="name of the task")
    parser.add_argument("-checkpoint", default=None, type=str,
                    help="checkpoint to resume for training")
    parser.add_argument('-num_epochs', default=100, type=int,
                    metavar='N', help='train_epochs')
    parser.add_argument('-seed', default=10, type=int,
                    metavar='N', help='random_seed')
    parser.add_argument("--split", action='store_true', help='split the train set or not')
    parser.add_argument("--param_only", action='store_true', help='only resume the parameters of the checkpoint')
    parser.add_argument("--resume_optimizer", action='store_true', help='resume the optimizer from the checkpoint')
    parser.add_argument("--resume_progress", action='store_true', help='resume the progress from the checkpoint')
    args = parser.parse_args()

    with open(args.train_config) as f:
        config = yaml.safe_load(f)
    model_config = config['model_config']
    # set_config = config['train_set_config']
    # valid_config = config["valid_set_config"]
    data_set_config = config['data_set_config']
    train_process_config = config['train_process_config'] if 'train_process_config' in config else None
    valid_process_config = config['valid_process_config'] if 'valid_process_config' in config else None

    set_config = copy(data_set_config)
    if train_process_config: set_config['trans_config'] = train_process_config

    valid_config = copy(data_set_config)
    if valid_process_config: valid_config['trans_config'] = valid_process_config

    train_config = config['train_config']
    opt_config = config["opti_config"]
    criterion_config = config["criterion_config"]
    module  =__import__(args.task_file,fromlist=[''])
    cls = getattr(module, args.task_name)
    if(args.num_gpu<0):
        cuda_id = -1
    else:
        cuda_id = 0
    task = cls(name=args.train_name, cuda_id=cuda_id, exp_path=args.exp_dir,
                    model_config=model_config, data_config=args.data_config, set_config=set_config, 
                    criterion_config=criterion_config, optim_config=opt_config, train_config=train_config, valid_config=valid_config,
                    other_config=None, random_seed=args.seed)
    task.task_init()

    if args.num_gpu>0:
        cuda_group = list(range(args.num_gpu))
        task.to_parallel(cuda_id, cuda_group)

    if not task.is_finish():
        print("task generate finish")
        exit()

    if args.checkpoint is not None:
        task.load_checkpoint(args.checkpoint, args.resume_optimizer, args.resume_progress)

    start_epoch = task.epoch + 1
        
    for i in range(start_epoch, args.num_epochs):
        task.train_epoch(i, args.split)
        task.do_valid()
        task.save_checkpoint(args.exp_dir+"/checkpoint."+str(i))

if __name__ == "__main__":
    main()