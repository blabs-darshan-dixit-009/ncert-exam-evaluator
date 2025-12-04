# models/training_callback.py

import logging
from transformers import TrainerCallback
from utils.training_progress import progress_manager

logger = logging.getLogger("exam_evaluator.training_callback")


class ProgressTrackingCallback(TrainerCallback):
    """
    Custom callback to track training progress in real-time.
    Updates progress manager at each epoch and step.
    """
    
    def __init__(self):
        super().__init__()
        self.current_epoch = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        print("\n" + "="*80)
        print("🚀 TRAINING STARTED")
        print("="*80)
        print(f"📊 Total Epochs: {args.num_train_epochs}")
        print(f"📦 Batch Size: {args.per_device_train_batch_size}")
        print(f"📈 Learning Rate: {args.learning_rate}")
        print(f"💾 Output Directory: {args.output_dir}")
        print("="*80 + "\n")
        
        logger.info("Training callback initialized", extra={
            "num_epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size
        })
        progress_manager.update_stage(
            "Training started",
            f"Starting training for {args.num_train_epochs} epochs"
        )
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        self.current_epoch = int(state.epoch) if state.epoch is not None else 0
        
        print("\n" + "─"*80)
        print(f"📖 EPOCH {self.current_epoch + 1}/{args.num_train_epochs} STARTED")
        print("─"*80)
        
        logger.info(f"Starting epoch {self.current_epoch + 1}/{args.num_train_epochs}")
        progress_manager.update_epoch(
            current_epoch=self.current_epoch + 1,
            total_epochs=int(args.num_train_epochs)
        )
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        # Get current loss
        loss = None
        if len(state.log_history) > 0:
            last_log = state.log_history[-1]
            loss = last_log.get('loss', None)
        
        # Calculate total steps per epoch
        total_steps_per_epoch = state.max_steps // args.num_train_epochs if args.num_train_epochs > 0 else state.max_steps
        current_step_in_epoch = state.global_step % total_steps_per_epoch if total_steps_per_epoch > 0 else state.global_step
        
        # Print progress every step
        if loss is not None:
            progress_bar = self._create_progress_bar(current_step_in_epoch, total_steps_per_epoch, 40)
            print(f"\r  Step {current_step_in_epoch:3d}/{total_steps_per_epoch} {progress_bar} Loss: {loss:.6f}", end='', flush=True)
        
        # Update progress manager less frequently
        if state.global_step % args.logging_steps == 0:
            progress_manager.update_step(
                current_step=current_step_in_epoch,
                total_steps=total_steps_per_epoch,
                loss=loss
            )
    
    def _create_progress_bar(self, current: int, total: int, width: int = 40) -> str:
        """Create a text-based progress bar"""
        if total == 0:
            return "[" + "─" * width + "]"
        
        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        percentage = int(100 * current / total)
        return f"[{bar}] {percentage:3d}%"
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        # Get epoch loss
        loss = None
        if len(state.log_history) > 0:
            last_log = state.log_history[-1]
            loss = last_log.get('loss', None)
        
        print()  # New line after progress bar
        print(f"✓ Epoch {self.current_epoch + 1} completed", end='')
        if loss is not None:
            print(f" - Loss: {loss:.6f}")
        else:
            print()
        
        logger.info(f"Completed epoch {self.current_epoch + 1}/{args.num_train_epochs}", extra={
            "epoch": self.current_epoch + 1,
            "loss": loss
        })
        
        progress_manager.update_epoch(
            current_epoch=self.current_epoch + 1,
            total_epochs=int(args.num_train_epochs),
            loss=loss
        )
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs and 'loss' in logs:
            # Detailed logging is now handled in on_step_end with progress bar
            logger.debug(f"Training log - Step: {state.global_step}, Loss: {logs['loss']:.6f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Model saved to: {args.output_dir}")
        print("="*80 + "\n")
        
        logger.info("Training completed successfully")
        progress_manager.update_stage(
            "Saving model",
            "Training completed, saving model and adapters..."
        )