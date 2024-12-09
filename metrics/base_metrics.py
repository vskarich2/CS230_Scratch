class BaseMetrics():
    # Order is create_table, update_table, log_table
    def __init__(self, o):
        self.o = o
        self.run = self.create_wandb_logs()
        self.run_table = self.create_run_table()

    def create_wandb_logs(self):
        pass

    def create_run_table(self):
        pass

    def create_epoch_table(self, epoch):
        pass

    def close_run_table(self):
        pass

    def close_epoch_table(self, epoch):
        pass

    def update_run_table(self, func):
        pass

    def update_epoch_table(self, epoch):
        pass