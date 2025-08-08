# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad, unnormalize_context
from train_test import train_epoch, test, analyze_and_save
from rag_utils import MolRAG
from torch.utils.tensorboard import SummaryWriter
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')

parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank for distributed training. Usually set by torch.distributed.launch or torchrun.')

# Latent Diffusion args
parser.add_argument('--train_diffusion', action='store_true', 
                    help='Train second stage LatentDiffusionModel model')
parser.add_argument('--ae_path', type=str, default=None,
                    help='Specify first stage model path')
parser.add_argument('--trainable_ae', action='store_true',
                    help='Train first stage AutoEncoder model')

# VAE args
parser.add_argument('--latent_nf', type=int, default=4,
                    help='number of latent features')
parser.add_argument('--kl_weight', type=float, default=0.01,
                    help='weight of KL term in ELBO')

parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
# RAG specific arguments
parser.add_argument('--use_rag', action='store_true', 
                    help='Whether to use Retrieval Augmented Generation (RAG).')
parser.add_argument('--rag_k', type=int, default=8, 
                    help='Number of retrieved items for RAG.')
parser.add_argument('--chemberta_path', type=str, default='DeepChem/ChemBERTa-77M-MTM', 
                    help='Path to ChemBERTa model for molecular embeddings.')
parser.add_argument('--rag_mol_emb_dim', type=int, default=384, 
                    help='Dimension of molecular embeddings used by RAGAggregator (e.g., ChemBERTa hidden size).')
parser.add_argument('--rag_num_prop', type=int, default=1, 
                    help='Number of properties for RAG (e.g., 12 for QM9). Should match len(conditioning).')
parser.add_argument('--rag_agg_dim', type=int, default=128, 
                    help='Aggregation dimension for RAG context.')
parser.add_argument('--batch_size_chemberta', type=int, default=32, 
                    help='Batch size for ChemBERTa embedding computation.')


args = parser.parse_args()

dataset_info = get_dataset_info(args.dataset, args.remove_h)

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
if "LOCAL_RANK" in os.environ:
    args.local_rank = int(os.environ["LOCAL_RANK"])
    print(f"[{os.getpid()}] Initializing DDP with local_rank: {args.local_rank}")
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    device = torch.device(f'cuda:{args.local_rank}')
    print(f"[{os.getpid()}] DDP initialized: rank {args.rank}/{args.world_size}, device {device}")
else:
    args.world_size = 1
    args.rank = 0
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Running without DDP: device {device}")

dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    # Store current RAG arguments to override loaded ones if they exist
    current_rag_args = {}
    for arg_name in ['use_rag', 'rag_k', 'chemberta_path', 'rag_mol_emb_dim', 'rag_num_prop', 'rag_agg_dim', 'batch_size_chemberta']:
        if hasattr(args, arg_name):
            current_rag_args[arg_name] = getattr(args, arg_name)
    
    current_ddp_args = {
        'local_rank': args.local_rank,
        'world_size': args.world_size,
        'rank': args.rank,
    }

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    # Restore RAG args to the loaded args object
    for k, v in current_rag_args.items():
        setattr(args, k, v)
    
    for k, v in current_ddp_args.items():
        setattr(args, k, v)

    print(args)

utils.create_folders(args)
# print(args)


# Wandb config
if args.rank == 0:
    if args.no_wandb:
        mode = 'disabled'
    else:
        mode = 'online' if args.online else 'offline'
    kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion_qm9', 'config': args,
            'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    log_dir = join('outputs', args.exp_name, 'tensorboard_logs')
    writer = SummaryWriter(log_dir)
else:
    writer = None

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

for partition, loader in dataloaders.items():
    if args.world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            loader.dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=(partition == 'train'), # Only shuffle train data
            drop_last=True # Ensure same number of samples for each process
        )
        # Recreate DataLoader with sampler
        dataloaders[partition] = torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size, # This is already PER_GPU_BATCH_SIZE
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True # Important for consistent batch sizes across processes
        )

rag_db = None
if args.use_rag:
    rag_db = MolRAG(args)

    rag_data_path = join(args.datadir, args.dataset + '_smiles_dataset.pt')
    rag_dataset = torch.load(rag_data_path)

    prop_keys = args.conditioning
    rag_db.build_db(rag_dataset, prop_keys)
data_dummy = next(iter(dataloaders['train']))


if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf

# Create Latent Diffusion Model or Audoencoder
if args.train_diffusion:
    model, nodes_dist, prop_dist, rag_aggregator = get_latent_diffusion(args, device, dataset_info, dataloaders['train'])
else:
    model, nodes_dist, prop_dist, rag_aggregator = get_autoencoder(args, device, dataset_info, dataloaders['train'])

if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)

if rag_aggregator is not None:
    rag_aggregator = rag_aggregator.to(device)

model = model.to(device)
optim = get_optim(args, model)
# print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def main():
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'flow.npy'), map_location=device)
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'), map_location=device)
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    if args.world_size > 1:
        model = DDP(model, device_ids=[args.local_rank])

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model.module if args.world_size > 1 else model)
        model_ema = model_ema.to(device)
        ema = flow_utils.EMA(args.ema_decay)
    else:
        ema = None
        model_ema = model

    best_nll_val = 1e8
    best_nll_test = 1e8
    best_epoch_val = -1

    global_step = args.start_epoch * len(dataloaders['train'])
    # Removed: scaler = GradScaler(enabled=args.use_amp) 

    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        if args.world_size > 1:
            dataloaders['train'].sampler.set_epoch(epoch)

        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist, 
                    rag_aggregator=rag_aggregator, rag_db=rag_db,
                    writer=writer, global_step=global_step) # Removed scaler=scaler
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        global_step += len(dataloaders['train']) 

        if args.rank == 0:
            if epoch % args.test_epochs == 0:
                current_model = model.module if args.world_size > 1 else model
                if isinstance(current_model, en_diffusion.EnVariationalDiffusion):
                    wandb.log(current_model.log_info(), commit=True)
                    if writer: writer.add_text("Model_Info", str(current_model.log_info()), global_step=epoch) # Example of logging text

                if not args.break_train_epoch and args.train_diffusion:
                    analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                    dataset_info=dataset_info, device=device,
                                    prop_dist=prop_dist, n_samples=args.n_stability_samples, 
                                    rag_aggregator=rag_aggregator, rag_db=rag_db)
                nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema,
                            partition='Val', device=device, dtype=dtype, 
                            nodes_dist=nodes_dist, property_norms=property_norms, 
                            rag_aggregator=rag_aggregator, rag_db=rag_db,
                            writer=writer)
                nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema,
                                partition='Test', device=device, dtype=dtype,
                                nodes_dist=nodes_dist, property_norms=property_norms,
                                rag_aggregator=rag_aggregator, rag_db=rag_db,
                                writer=writer)
                
                wandb.log({"Val loss ": nll_val, "epoch": epoch}, commit=True)
                wandb.log({"Test loss ": nll_test, "epoch": epoch}, commit=True)
                if writer:
                    writer.add_scalar("NLL/Validation", nll_val, global_step=epoch)
                    writer.add_scalar("NLL/Test", nll_test, global_step=epoch)

                if nll_val < best_nll_val:
                    best_nll_val = nll_val
                    best_nll_test = nll_test
                    best_epoch_val = epoch

                    if writer:
                        writer.add_scalar("NLL/Best_Validation", best_nll_val, global_step=epoch)
                        writer.add_scalar("NLL/Best_Test", best_nll_test, global_step=epoch)
                        writer.add_scalar("NLL/Best_Epoch", best_epoch_val, global_step=epoch)

                    if args.save_model:
                        args.current_epoch = epoch + 1
                        utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                        # Save the unwrapped module
                        utils.save_model(model.module if args.world_size > 1 else model, 'outputs/%s/generative_model.npy' % args.exp_name)
                        if args.ema_decay > 0:
                            utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                        with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                            pickle.dump(args, f)

                print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
                print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
                wandb.log({"Val loss ": nll_val}, commit=True)
                wandb.log({"Test loss ": nll_test}, commit=True)
                wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)
        
        if args.world_size > 1:
            dist.barrier() 
    
    if writer and args.rank == 0:
        writer.close()
    
    if args.world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
