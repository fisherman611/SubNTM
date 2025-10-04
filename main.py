from utils import config, log, miscellaneous, seed
import os
import numpy as np
import basic_trainer
from SubNTM.SubNTM import SubNTM
import evaluations
import datasethandler
import scipy
import torch
from types import SimpleNamespace
import multiprocessing as mp

RESULT_DIR = 'results'
DATA_DIR = 'datasets'

if __name__ == "__main__":
    # try:
    #     mp.set_start_method("spawn", force=True)
    # except RuntimeError:
    #     pass
    parser = config.new_parser()
    config.add_dataset_argument(parser)
    config.add_model_argument(parser)
    config.add_logging_argument(parser)
    config.add_training_argument(parser)
    config.add_eval_argument(parser)
    args = parser.parse_args()
    
    prj = args.wandb_prj if args.wandb_prj else 'baselines'

    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR + "/" + "SubNTM" + "/" +str(args.dataset), current_time)
    miscellaneous.create_folder_if_not_exist(current_run_dir)

    config.save_config(args, os.path.join(current_run_dir, 'config.txt'))
    seed.seedEverything(args.seed)
    print(args)
    
    logger = log.setup_logger(
        'main', os.path.join(current_run_dir, 'main.log'))
    read_labels = True

    # load a preprocessed dataset
    dataset = datasethandler.BasicDatasetHandler(
        os.path.join(DATA_DIR, args.dataset), device=args.device, read_labels=read_labels,
        as_tensor=True, contextual_embed=False)

    # create a model
    pretrainWE = scipy.sparse.load_npz(os.path.join(
        DATA_DIR, args.dataset, "word_embeddings.npz")).toarray()
    
    sub_args = SimpleNamespace(
        vocab_size=dataset.vocab_size,
        en1_units=args.en1_units,
        dropout=args.dropout,
        embed_size=args.embed_size,
        num_topic=args.num_topic,
        adapter_alpha=args.adapter_alpha,
        beta_temp=args.beta_temp,
        tau=args.tau,
        weight_loss_ECR=args.weight_loss_ECR,
        sinkhorn_alpha=args.sinkhorn_alpha,
        sinkhorn_max_iter=args.sinkhorn_max_iter,
        augment_coef=args.augment_coef,
        word_embeddings=pretrainWE,
        lambda_doc=args.lambda_doc
    )
    model = SubNTM(sub_args).cuda() if args.device == 'cuda' else SubNTM(sub_args)
    # model = torch.compile(model, mode="max-autotune")

    # create a trainer
    trainer = basic_trainer.BasicTrainer(model, epochs=args.epochs,
                        learning_rate=args.lr,
                        batch_size=args.batch_size,
                        lr_scheduler=args.lr_scheduler,
                        lr_step_size=args.lr_step_size,
                        device=args.device)


    # train the model

    trainer.train(dataset)
    # save beta, theta and top words
    beta = trainer.save_beta(current_run_dir)
    train_theta, test_theta = trainer.save_theta(dataset, current_run_dir)
    
    top_words_15 = trainer.save_top_words(
        dataset.vocab, 15, current_run_dir)

    # argmax of train and test theta
    # train_theta_argmax = train_theta.argmax(axis=1)
    # test_theta_argmax = test_theta.argmax(axis=1) 
    train_theta_argmax = train_theta.argmax(axis=1)
    unique_elements, counts = np.unique(train_theta_argmax, return_counts=True)
    print(f'train theta argmax: {unique_elements, counts}')
    logger.info(f'train theta argmax: {unique_elements, counts}')
    test_theta_argmax = test_theta.argmax(axis=1)
    unique_elements, counts = np.unique(test_theta_argmax, return_counts=True)
    print(f'test theta argmax: {unique_elements, counts}')
    logger.info(f'test theta argmax: {unique_elements, counts}')       

    # evaluating clustering
    if read_labels:
        clustering_results = evaluations.evaluate_clustering(
            test_theta, dataset.test_labels)
        print(f"NMI: ", clustering_results['NMI'])
        print(f'Purity: ', clustering_results['Purity'])

    # evaluate classification
    if read_labels:
        classification_results = evaluations.evaluate_classification(
            train_theta, test_theta, dataset.train_labels, dataset.test_labels, tune=args.tune_SVM)
        print(f"Accuracy: ", classification_results['acc'])
        print(f"Macro-f1", classification_results['macro-F1'])

    # IRBO
    IRBO_15 = evaluations.irbo.buubyyboo_dth([top_words.split() for top_words in top_words_15], topk=15)
    print(f"IRBO_15: {IRBO_15:.5f}")

    # TC
    TC_15_list, TC_15 = evaluations.topic_coherence.TC_on_wikipedia(
        os.path.join(current_run_dir, 'top_words_15.txt'))
    print(f"TC_15: {TC_15:.5f}")