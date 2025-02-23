# filepath: ../src/print_epoch_callback.py
import lightning as L

class PrintEpochCallback(L.Callback):
    def on_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print(f"Epoch {trainer.current_epoch} finished.")