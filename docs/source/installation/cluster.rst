Training models in parallel
===========================

*Please note the following steps are specific when working on the Nikhef Stoomboot computing cluster, running
CentOS 7 Linux distribution as of writing this. Other setups may require an alternate script configuration.*

Training models on different replicas can be done in parallel, for example on a computing cluster.
EELSFitter supports parallel training provided the user specifies some parameters.
These are the number of batch jobs ``n_batches`` that are sent to the cluster (same value for all batches),
the number of replicas ``n_replica`` that are trained per batch (same value for all batches),
and an index corresponding to a particular batch ``n_batch_of_replica`` (different value for all batches).
These can be for example be passed by command line to the script.
If you are running the code on your own machine on a single core ``n_batches=1`` and ``n_batch_of_replica=1``.

An example setup for submitting code to a cluster is shown below.
First a bash script executes commands to submit tasks to a job scheduler.

.. code-block:: bash

    #!/bin/bash
    pbs_file=/path/to/pbs_file.pbs

    path_to_image="/path/to/dm4_file.dm4"
    path_to_models="/path/to/output of models/"

    n_batches=100
    for n_batch_of_replica in `seq 1 $n_batches`; do
        <cluster_specific_submission_code> ARG=$path_image,ARG2=$path_models,ARG3=$n_batch_of_replica,ARG4=$n_batches $pbs_file
    done

A .pbs file specifies where the Python installation is located such that the system can actually execute the code.

.. code-block::

    source /path/to/miniconda3/etc/profile.d/conda.sh
    conda activate <environmentname>
    python /path/to/python_file.py ${ARG} ${ARG2} ${ARG3} ${ARG4}

Finally the Python file contains that which you want to execute.

.. code-block:: python

    import sys
    import EELSFitter as ef
    
    path_to_image = sys.argv[1]
    path_to_models = sys.argv[2]
    n_batch_of_replica = int(sys.argv[3])
    n_batches = int(sys.argv[4])

    im = ef.SpectralImage.load_data(path_to_image)
    im.train_zlp_models(n_clusters=n_clusters,
                        seed=seed,
                        based_on=based_on,
                        n_replica=n_replica,
                        n_epochs=n_epochs,
                        n_batch_of_replica=n_batch_of_replica,
                        n_batches=n_batches,
                        shift_de1=shift_dE1,
                        shift_de2=shift_dE2,
                        regularisation_constant=regularisation_constant,
                        path_to_models=path_to_models,
                        signal_type=signal_type)
