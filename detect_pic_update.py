import datetime
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from for_qt_test_02 import get_in


class MyEventHandler(FileSystemEventHandler):
    def __init__(self):
        FileSystemEventHandler.__init__(self)

    def on_any_event(self, event):
        print("-----")
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))


    # 新建
    def on_created(self, event):
        event = event
        if event.is_directory:
            # def return_info(src_path):
            #     return get_in(event.src_path)
            # return_info(event.src_path)
            print(event.src_path)
            print("目录 created:{file_path}".format(file_path=event.src_path))
        else:
            print("文件 created:{file_path}".format(file_path=event.src_path))





if __name__ == '__main__':
    path = r"./save_pic"

    myEventHandler = MyEventHandler()

    # 观察者
    observer = Observer()

    # recursive:True 递归的检测文件夹下所有文件变化。
    observer.schedule(myEventHandler, path, recursive=True)

    # 观察线程，非阻塞式的。
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()