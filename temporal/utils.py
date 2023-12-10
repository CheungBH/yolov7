

class TemporalQueue:
    def __init__(self, queue_size=5, queue_interval=2):
        self.queue = []
        self.queue_size = queue_size
        self.queue_interval = queue_interval
        self.feed = False
        self.interval_cnt = 0

    def update(self, data):
        self.queue.append(data)
        if len(self.queue) > self.queue_size:
            self.queue.pop(0)

    def determine_feed(self):
        if len(self.queue) == self.queue_size:
            if self.interval_cnt == self.queue_interval-1:
                self.interval_cnt = 0
                self.feed = True
            else:
                self.interval_cnt += 1
                self.feed = False
        else:
            self.feed = False

    def get_data(self):
        if self.feed:
            return self.queue
        else:
            return None

    def print_queue(self):
        print(self.queue)



