def export_model(config, Shuffles=[1]):
    """

    Exports model trained at specified shuffles
    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    Shuffles: list, optional
        List of integers specifying the shuffle indices of the training dataset. The default is [1]
    """
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    from deeplabcut.pose_estimation_tensorflow.nnet import predict
    from deeplabcut.utils import auxiliaryfunctions
    import tensorflow as tf
    from pathlib import Path
    import numpy as np
    import os

    tf.reset_default_graph()

    # Loading config from pose_cfg
    cfg = auxiliaryfunctions.read_config(config)
    TrainingFractions = cfg['TrainingFraction']

    # Make folder for export data
    auxiliaryfunctions.attempttomakefolder(os.path.join(cfg['project_path'], 'exported_model'))
    
    for shuffle in Shuffles:
        for trainFraction in TrainingFractions:
            modelfolder = str(auxiliaryfunctions.GetModelFolder(trainFraction, shuffle, cfg))
            modelfolder = os.path.join(cfg['project_path'], modelfolder)
            path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'

            try:
                dlc_cfg = load_config(str(path_test_config))
            except FileNotFoundError:
                raise FileNotFoundError('It seem the model for shuffle {} and trainFraction {} does not exis'.format(shuffle, trainFraction) )
            
            # Create folder structure to store exported models 
            exportfolder = os.path.join(cfg['project_path'], str(auxiliaryfunctions.GetExportFolder(trainFraction, shuffle, cfg)))
            auxiliaryfunctions.attempttomakefolder(exportfolder, recursive=True)

            # Check which snapshots are available and sort them by # iterations
            Snapshots = np.array([fn.split('.')[0] for fn in os.listdir(os.path.join(modelfolder, 'train')) if 'index' in fn])
            try:
                # Check if any were found
                Snapshots[0]
            except IndexError:
                raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle {} and trainFraction {} is not trained.\nPlease train it before evaluating.\nUse the function 'train_network' to do so.".format(shuffle,trainFraction))
            increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
            # Choosing last snapshot 
            snapshot = Snapshots[increasing_indices[-1]]

            # Setting weights to corresponding snapshot
            dlc_cfg['init_weights'] = os.path.join(modelfolder, 'train', snapshot)

            # Read how many training siterations that corresponds to
            trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]
            # Name for deeplabcut net (based on its parameters)
            DLCscorer, _ = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations)

            # Load model
            sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)

            # Save model
            saver = tf.train.Saver()
            savepath = os.path.join(exportfolder, 'DLCscorer')
            saver.save(sess, savepath)
            print('Saved model to {}!'.format(savepath))
            # Close session
            sess.close()
