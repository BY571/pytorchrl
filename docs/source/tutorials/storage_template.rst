Storage template
================

As reference examples, see `OnPolicyBuffer <https://github.com/PyTorchRL/pytorchrl/blob/master/pytorchrl/core/storages/on_policy/on_policy_buffer.py>`_ and `ReplayBuffer <https://github.com/PyTorchRL/pytorchrl/blob/master/pytorchrl/core/storages/off_policy/replay_buffer.py>`_

.. code-block:: python

    from abc import ABC, abstractmethod

    class Storage(ABC):
        """
        Base class for all storage components. It should serve as a template to
        create new Storage classes with new or extended features.
        """

        @classmethod
        @abstractmethod
        def create_factory(cls, *args):
            """Returns a function to create new Storage instances"""
            raise NotImplementedError

        @abstractmethod
        def init_tensors(self, sample, *args):
            """
            Lazy initialization of data tensors from a sample.

            Parameters
            ----------
            sample : dict
                Data sample (containing all tensors of an environment transition)
            """
            raise NotImplementedError

        @abstractmethod
        def get_data(self, data_to_cpu=False, *args):
            """
            Return currently stored data. If data_to_cpu, make sure to move
            data tensors to cpu memory.
            """
            raise NotImplementedError

        @abstractmethod
        def add_data(self, new_data, *args):
            """
            Replace currently stored data.

            Parameters
            ----------
            new_data : dict
                Dictionary of env transition samples to replace self.data with.
            """
            raise NotImplementedError

        @abstractmethod
        def reset(self, *args):
            """Set class counters to zero and remove stored data"""
            raise NotImplementedError

        @abstractmethod
        def insert(self, sample, *args):
            """
            Store new transition sample.

            Parameters
            ----------
            sample : dict
                Data sample (containing all tensors of an environment transition)
            """
            raise NotImplementedError

        @abstractmethod
        def before_update(self, actor_critic, algo, *args):
            """
            Before updating actor policy model, compute returns and advantages.

            Parameters
            ----------
            actor_critic : ActorCritic
                An actor_critic class instance.
            algo : an algorithm class
                An algorithm class instance.
            """
            raise NotImplementedError

        @abstractmethod
        def after_update(self, *args):
            """After updating actor policy model, make sure self.step is at 0."""
            raise NotImplementedError

        @abstractmethod
        def generate_batches(self, num_mini_batch, mini_batch_size, num_epochs=1, recurrent_ac=False, *args):
            """
            Returns a batch iterator to update actor critic.

            Parameters
            ----------
            num_mini_batch : int
               Number mini batches per epoch.
            mini_batch_size : int
                Number of samples contained in each mini batch.
            num_epochs : int
                Number of epochs.
            recurrent_ac : bool
                Whether actor critic policy is a RNN or not.
            shuffle : bool
                Whether to shuffle collected data or generate sequential

            Yields
            ______
            batch : dict
                Generated data batches.
            """
            raise NotImplementedError

