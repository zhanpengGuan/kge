import time

# 定义一个装饰器函数
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time} 秒")
        return result
    return wrapper

# 使用装饰器来装饰函数
@timing_decorator
def some_function():
    # 模拟一个耗时操作
    time.sleep(2)
    print("函数执行完毕")

some_function()
