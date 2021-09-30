# CloudShield

Implementation of anomaly detection in the cloud of the following paper:

[He, Zecheng, and Ruby B. Lee. "CloudShield: Real-time Anomaly Detection in the Cloud." arXiv preprint arXiv:2108.08977 (2021).](https://arxiv.org/abs/2108.08977)

### Data

Hardware performance counter data of the attacks (especially the speculative execution attacks like Spectre and Meltdown) and benign programs:

https://drive.google.com/drive/folders/1Gz-gji9nj9TkLAqCrgwo_4doGhMmUYlh?usp=sharing

Download and put them under CloudShield/data/

### Code structure and usage

    attack/:
        # implementations of the following attacks
        # L1 prime-probe
        # L3 prime-probe
        # Flush-reload
        # Flush-flush
        # Spectre-v1
        # Spectre-v2
        # Spectre-v3 (Meltdown)
        # Spectre-SSB

    bg_program/:
        # benign programs
        # SPEC 2006 benchmarks
        # Gpg-RSA

    cloud_workload/:
        # Representative cloud workloads
        # Database (mysql)
        # MapReduce
        # ML training (pytorch)
        # Web server (nginx)
        # Stream server (ffserver)

    detector/:
        # Anomaly detection scripts
        # pretrain_model.ipynb: pretrain the model
        # kernel_density_estimation.ipynb: KDE for anomaly detection (step 1)
        # known_attack_detection.ipynb: known attack detection (step 2)
        # benign_program_detection.ipynb: benign program detection (step 2)

    perf/:
        # Scripts for collecting HPCs
        # Please first run the corresponding cloud workload in cloud_workload/, attacks in attack/ and benign programs in bg_program/.

        # Example command to run the HPC collection
        python3 data_collection.py --core 3 --us 10000 --n_readings 12000.
        # It collects HPCs on core #3, interval 10000 us (10ms) and total 12000 readings.

### Reference

You are encouraged to refer the following paper:

    @article{he2021cloudshield,
        title={CloudShield: Real-time Anomaly Detection in the Cloud},
        author={He, Zecheng and Lee, Ruby B},
        journal={arXiv preprint arXiv:2108.08977},
        year={2021}
    }
