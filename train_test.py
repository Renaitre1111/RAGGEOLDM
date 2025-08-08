# train_test.py
import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import torch
import torch.distributed as dist


def train_epoch(args, loader, epoch, model, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist, rag_aggregator=None, rag_db=None, writer=None, global_step=0): # Removed scaler 参数
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        context = None
        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)

        adaln_ctx = None
        if args.use_rag:
            # Note: For AMP, ensure these are already on device and correct dtype (float32 is fine, autocast handles it) - AMP removed
            target_norm_raw = context[:, :1, :].squeeze(1) # context is (bs, n_nodes, num_props), this becomes (bs, num_props)
            # target_unnorm is not used in the model forward, so its dtype doesn't directly impact AMP here. - AMP removed
            target_unnorm_ret = qm9utils.unnormalize_context(context[:, :1, :], args.conditioning, property_norms)[:, 0, :].cpu().numpy()
            ret_mol_embs, ret_prop_vals, ret_dist_to_targs, _ = rag_db.retrieve_batch(target_unnorm_ret, k=args.rag_k)
            
            ret_mol_embs = torch.from_numpy(ret_mol_embs).to(device, dtype)
            ret_prop_vals = torch.from_numpy(ret_prop_vals).to(device, dtype)
            ret_dist_to_targs = torch.from_numpy(ret_dist_to_targs).to(device, dtype)

            # Pass the raw target_norm for RAGAggregator as it expects (bs, num_props)
            adaln_ctx = rag_aggregator(target_norm_raw, ret_mol_embs, ret_prop_vals, ret_dist_to_targs) # (bs, agg_dim)

            adaln_ctx = adaln_ctx.unsqueeze(1).repeat(1, x.size(1), 1) * node_mask  # (bs, n_nodes, agg_dim)

        optim.zero_grad()

        # Removed: with autocast(enabled=args.use_amp):
        # transform batch through flow
        nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model, nodes_dist,
                                                                x, h, node_mask, edge_mask, context, adaln_ctx=adaln_ctx)
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        
        # Removed: if args.use_amp: scaler.scale(loss).backward()
        # Replaced with:
        loss.backward()

        if args.clip_grad:
            # Removed: if args.use_amp: scaler.unscale_(optim)
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        # Removed: if args.use_amp: scaler.step(optim); scaler.update()
        # Replaced with:
        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model.module if args.world_size > 1 else model)

        current_global_step = global_step + i 

        nll_epoch.append(nll.item())
        
        if args.rank == 0:
            if i % args.n_report_steps == 0:
                print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                    f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                    f"RegTerm: {reg_term.item():.1f}, "
                    f"GradNorm: {grad_norm:.1f}")

            if writer:
                writer.add_scalar("NLL/Train_Batch", nll.item(), global_step=current_global_step)

            if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0) and args.train_diffusion:
                start = time.time()
                if len(args.conditioning) > 0:
                    save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch, 
                                                rag_aggregator=rag_aggregator, rag_db=rag_db)
                save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
                                    batch_id=str(i), rag_aggregator=rag_aggregator, rag_db=rag_db)
                sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
                                                prop_dist, epoch=epoch, rag_aggregator=rag_aggregator, rag_db=rag_db)
                print(f'Sampling took {time.time() - start:.2f} seconds')

        if args.break_train_epoch:
            break
    
    if args.world_size > 1:
        nll_epoch = torch.tensor(nll_epoch, device=device, dtype=dtype)
        gathered_nll_epoch = [torch.zeros_like(nll_epoch) for _ in range(args.world_size)]
        dist.all_gather(gathered_nll_epoch, nll_epoch)
        all_nlls = torch.cat(gathered_nll_epoch).cpu().numpy()
        mean_nll = np.mean(all_nlls)
    else:
        mean_nll = np.mean(nll_epoch)

    if args.rank == 0:
        wandb.log({"Train Epoch NLL": mean_nll}, commit=False)
        if writer:
            writer.add_scalar("NLL/Train_Epoch", mean_nll, global_step=epoch)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test', rag_aggregator=None, rag_db=None, writer=None):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            adaln_ctx = None

            if args.use_rag:
                # Note: For AMP, ensure these are already on device and correct dtype (float32 is fine, autocast handles it) - AMP removed
                target_norm_raw = context[:, :1, :].squeeze(1) # context is (bs, n_nodes, num_props), this becomes (bs, num_props)
                # target_unnorm is not used in the model forward, so its dtype doesn't directly impact AMP here. - AMP removed
                target_unnorm_ret = qm9utils.unnormalize_context(context[:, :1, :], args.conditioning, property_norms)[:, 0, :].cpu().numpy()
                ret_mol_embs, ret_prop_vals, ret_dist_to_targs, _ = rag_db.retrieve_batch(target_unnorm_ret, k=args.rag_k)
                
                ret_mol_embs = torch.from_numpy(ret_mol_embs).to(device, dtype)
                ret_prop_vals = torch.from_numpy(ret_prop_vals).to(device, dtype)
                ret_dist_to_targs = torch.from_numpy(ret_dist_to_targs).to(device, dtype)

                # Pass the raw target_norm for RAGAggregator as it expects (bs, num_props)
                # `aggregate` method is used here, ensure it has the same signature
                adaln_ctx = rag_aggregator(target_norm_raw, ret_mol_embs, ret_prop_vals, ret_dist_to_targs) # (bs, agg_dim)

                adaln_ctx = adaln_ctx.unsqueeze(1).repeat(1, x.size(1), 1) * node_mask  # (bs, n_nodes, agg_dim)

            # Removed: with autocast(enabled=args.use_amp):
            # transform batch through flow
            nll, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                    node_mask, edge_mask, context, adaln_ctx=adaln_ctx)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if args.rank == 0:
                if i % args.n_report_steps == 0:
                    print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                        f"NLL: {nll_epoch/n_samples:.2f}")
        if args.world_size > 1:
            # Convert accumulated NLL and samples to tensors
            total_nll = torch.tensor(nll_epoch, device=device, dtype=dtype)
            total_samples = torch.tensor(n_samples, device=device, dtype=torch.int)

            # Reduce sum across all processes
            dist.all_reduce(total_nll, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # Compute global mean NLL
            global_mean_nll = total_nll.item() / total_samples.item()
        else:
            global_mean_nll = nll_epoch / n_samples

        if args.rank == 0:
            if writer:
                writer.add_scalar(f"NLL/{partition}_Epoch", global_mean_nll, global_step=epoch)

    return global_mean_nll


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id='', rag_aggregator=None, rag_db=None):
    model_unwrapped = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model_unwrapped,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist,
                                       rag_aggregator=rag_aggregator, rag_db=rag_db)
    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id='', rag_aggregator=None, rag_db=None):
    model_unwrapped = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        # Removed: with autocast(enabled=args.use_amp):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model_unwrapped, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info,
                                                rag_aggregator=rag_aggregator, rag_db=rag_db)
        
        print(f"Generated molecule: Positions {x[:-1, :, :]}")


def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100, rag_aggregator=None, rag_db=None):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    model_unwrapped = model_sample.module if isinstance(model_sample, torch.nn.parallel.DistributedDataParallel) else model_sample

    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        # Removed: with autocast(enabled=args.use_amp):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model_unwrapped, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample, rag_aggregator=rag_aggregator, rag_db=rag_db)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0,
                                rag_aggregator=None, rag_db=None):
    model_unwrapped = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model_unwrapped, dataset_info, prop_dist,
                                                               rag_aggregator=rag_aggregator, rag_db=rag_db)

    return one_hot, charges, x
