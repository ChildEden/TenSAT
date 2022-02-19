import time


def print_log(msg, f=None):
    log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    if f is not None:
        if type(f) == list:
            for sf in f:
                print(f'{log_time}: {msg}', file=sf, flush=True)
        else:
            print(f'{log_time}: {msg}', file=f, flush=True)
    print(f'{log_time}: {msg}')
