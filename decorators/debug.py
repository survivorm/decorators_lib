import functools
import linecache
import logging
import multiprocessing
import os
import signal
import sys
import time
import warnings

try:
    import redis_lock
except:
    redis_lock = None

from hasoffers.core.util.helpers.redis_actions import UTILITY, RedisSession
from hasoffers.core.util.common_logger import get_current_root_logger

_redis = RedisSession(UTILITY)

LOG_PRINTER = default_logger = get_current_root_logger()


########################################################################################################################

class ContextDecorator(object):
    def __call__(self, f, *args, **kwargs):
        @functools.wraps(f)
        def decorated(*args, **kwds):
            self.dec_f = f
            with self as _ctx:
                if self.push_context:
                    return f(_ctx, *args, **kwds)
                return f(*args, **kwds)

        return decorated

    @property
    def name(self):
        if hasattr(self, '_name'):
            return self._name
        elif hasattr(self, 'dec_f'):
            return self.dec_f.__name__
        else:
            return ''


class track_entry_and_exit(ContextDecorator):
    def __init__(self, name, logger=default_logger):
        self._name = name
        self.logger = logger

    def __enter__(self):
        self.logger.info('Entering: {}'.format(self.name))

    def __exit__(self, exc_type, exc, exc_tb):
        self.logger.info('Exiting: {}'.format(self.name))


class track_exec_time(ContextDecorator):
    def __init__(self, logger=default_logger):
        self.logger = logger
        self.started = time.time()

    def __enter__(self):
        self.logger.info('Entering: {name} at {time}'.format(name=self.name, time=time.ctime()))

    def __exit__(self, exc_type, exc, exc_tb):
        self.logger.info('Exiting:  {name} at {time}. Elapsed {elapsed:.2}'.format(name=self.name, time=time.ctime(),
                                                                                   elapsed=time.time() - self.started))


########################################################################################################################

# from PythonDecoratorLibrary:
def addto(instance):
    # Easy adding methods to a class instance
    # @addto(foo)
    # def print_x(self):
    # print self.x

    # foo.print_x() would print foo.x
    def decorator(f):
        import types
        f = types.MethodType(f, instance, instance.__class__)
        setattr(instance, f.func_name, f)
        return f

    return decorator


########################################################################################################################


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    @other_decorators_must_be_upper
    @deprecated
    def my_func():
        pass

    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn_explicit(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            filename=func.func_code.co_filename,
            lineno=func.func_code.co_firstlineno + 1
        )
        return func(*args, **kwargs)

    return new_func


########################################################################################################################


def trace(f):
    """
    Line Tracing Individual Functions
    """

    def globaltrace(frame, why, arg):
        if why == "call":
            return localtrace
        return None

    def localtrace(frame, why, arg):
        if why == "line":
            # record the file name and line number of every trace
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno

            bname = os.path.basename(filename)
            print "{}({}): {}".format(bname,
                                      lineno,
                                      linecache.getline(filename, lineno)),
        return localtrace

    @functools.wraps(f)
    def _f(*args, **kwds):
        sys.settrace(globaltrace)
        result = f(*args, **kwds)
        sys.settrace(None)
        return result

    return _f


########################################################################################################################


'''
One of three degrees of enforcement may be specified by passing
the 'debug' keyword argument to the decorator:
    0 -- NONE:   No type-checking. Decorators disabled.
 #!python
-- MEDIUM: Print warning message to stderr. (Default)
    2 -- STRONG: Raise TypeError with message.
If 'debug' is not passed to the decorator, the default level is used.

Example usage:
    >>> NONE, MEDIUM, STRONG = 0, 1, 2
    >>>
    >>> @accepts(int, int, int)
    ... @returns(float)
    ... def average(x, y, z):
    ...     return (x + y + z) / 2
    ...
    >>> average(5.5, 10, 15.0)
    TypeWarning:  'average' method accepts (int, int, int), but was given
    (float, int, float)
    15.25
    >>> average(5, 10, 15)
    TypeWarning:  'average' method returns (float), but result is (int)
    15

Needed to cast params as floats in function def (or simply divide by 2.0).

    >>> TYPE_CHECK = STRONG
    >>> @accepts(int, debug=TYPE_CHECK)
    ... @returns(int, debug=TYPE_CHECK)
    ... def fib(n):
    ...     if n in (0, 1): return n
    ...     return fib(n-1) + fib(n-2)
    ...
    >>> fib(5.3)
    Traceback (most recent call last):
      ...
    TypeError: 'fib' method accepts (int), but was given (float)

'''

NONE, MEDIUM, STRONG = 0, 1, 2


def accepts(*types, **kw):
    """Function decorator. Checks decorated function's arguments are
    of the expected types.

    Parameters:
    types -- The expected types of the inputs to the decorated function.
             Must specify type for each parameter.
    kw    -- Optional specification of 'debug' level (this is the only valid
             keyword argument, no other should be given).
             debug = ( 0 | 1 | 2 )

    """
    if not kw:
        # default level: MEDIUM
        debug = 1
    else:
        debug = kw['debug']
    try:
        def decorator(f):
            @functools.wraps(f)
            def newf(*args):
                if debug is 0:
                    return f(*args)
                assert len(args) == len(types)
                argtypes = tuple(map(type, args))
                if argtypes != types:
                    msg = info(f.__name__, types, argtypes, 0)
                    if debug is 1:
                        print >> sys.stderr, 'TypeWarning: ', msg
                    elif debug is 2:
                        raise TypeError, msg
                return f(*args)

            return newf

        return decorator
    except KeyError, key:
        raise KeyError, key + "is not a valid keyword argument"
    except TypeError, msg:
        raise TypeError, msg


def returns(ret_type, **kw):
    """Function decorator. Checks decorated function's return value
    is of the expected type.

    Parameters:
    ret_type -- The expected type of the decorated function's return value.
                Must specify type for each parameter.
    kw       -- Optional specification of 'debug' level (this is the only valid
                keyword argument, no other should be given).
                debug=(0 | 1 | 2)
    """
    try:
        if not kw:
            # default level: MEDIUM
            debug = 1
        else:
            debug = kw['debug']

        def decorator(f):
            @functools.wraps(f)
            def newf(*args):
                result = f(*args)
                if debug is 0:
                    return result
                res_type = type(result)
                if res_type != ret_type:
                    msg = info(f.__name__, (ret_type,), (res_type,), 1)
                    if debug is 1:
                        print >> sys.stderr, 'TypeWarning: ', msg
                    elif debug is 2:
                        raise TypeError, msg
                return result

            return newf

        return decorator
    except KeyError, key:
        raise KeyError, key + "is not a valid keyword argument"
    except TypeError, msg:
        raise TypeError, msg


def info(fname, expected, actual, flag):
    """Convenience function returns nicely formatted error/warning msg."""
    format = lambda types: ', '.join([str(t).split("'")[1] for t in types])
    expected, actual = format(expected), format(actual)
    msg = "'{}' method ".format(fname) \
          + ("accepts", "returns")[flag] + " ({}), but ".format(expected) \
          + ("was given", "result is")[flag] + " ({})".format(actual)
    return msg


########################################################################################################################


class LogPrinter:
    """LogPrinter class which serves to emulates a file object and logs
       whatever it gets sent to a Logger object at the INFO level."""

    def __init__(self, logger=default_logger):
        """Grabs the specific logger to use for logprinting."""
        self.ilogger = logger
        il = self.ilogger
        logging.basicConfig()
        il.setLevel(logging.INFO)

    def write(self, text):
        """Logs written output to a specific logger"""
        self.ilogger.info(text)


def logprintinfo(func):
    """Wraps a method so that any calls made to print get logged instead"""

    @functools.wraps(func)
    def pwrapper(*arg, **kwargs):
        stdobak = sys.stdout
        lpinstance = LogPrinter()
        sys.stdout = lpinstance
        try:
            return func(*arg, **kwargs)
        finally:
            sys.stdout = stdobak

    return pwrapper


########################################################################################################################

class TimeoutError(Exception): pass


def timeout(seconds, error_message='Function call timed out'):
    def decorated(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return functools.wraps(func)(wrapper)

    return decorated


########################################################################################################################

def unchanged(func):
    "This decorator doesn't add any behavior"
    return func


def disabled(func):
    "This decorator disables the provided function, and does nothing"

    def empty_func(*args, **kargs):
        pass

    return empty_func


# define this as equivalent to unchanged, for nice symmetry with disabled
enabled = unchanged


#
#  Sample use
#

# GLOBAL_ENABLE_FLAG = True

# state = enabled if GLOBAL_ENABLE_FLAG else disabled
# @state
# def special_function_foo():
#   print "function was enabled"

########################################################################################################################


class redlock(ContextDecorator):
    """
        Redis locked decorator/context manager
        used either via
        @redlock(name='lock_name', raise_blocked=True)
        def smth():
            ...

        or via
        with redlock(name='lock_name', raise_blocked=True):
            ...


        params:
        @param conn=_redis - StrictRedis instance
        @param name=str(time.time()) - key name
        @param blocking=False
        @param raise_blocked=False - raise if lock is not acquired
        @param suppress_inner_lock_ex=False - raise if lock is not acquired somethere inside ctx
    """

    class Locked(Exception):
        def __init__(self, *args, **kwargs):
            self._id = kwargs.pop('_id')
            super(redlock.Locked, self).__init__(*args, **kwargs)

        def raised_by_me(self, _id):
            print _id, self._id
            return _id == self._id

    def __init__(self, conn=_redis, name=str(time.time()), blocking=False, raise_blocked=False,
                 suppress_inner_lock_ex=False, timeout=None, **kwargs):
        self.lock = redis_lock.Lock(conn, name, **kwargs)
        self._blocking = blocking  # block untill lock released
        self.timeout = timeout
        self.raise_blocked = raise_blocked  # Raise if ca't aquire lock myself
        self.suppress_inner_lock_ex = suppress_inner_lock_ex  # Suppress inner exceptions which were unable to get lock
        self.kwargs = kwargs

    def __enter__(self):
        kwa = {'blocking': self._blocking}
        if self.timeout:
            kwa['timeout'] = self.timeout
        if self.lock.acquire(**kwa):
            pass
        else:
            try:
                raise self.Locked('Can\'t acquire lock in {name}'.format(name=self.name), _id=id(self))
            except:
                # Swallow exception if __exit__ returns a True value
                if self.__exit__(*sys.exc_info()):
                    pass
                else:
                    raise

    def __exit__(self, exc_type, exc, exc_tb):
        self.close()
        if exc_type == self.Locked:
            if not self.raise_blocked and exc.raised_by_me(id(self)):
                print('Lock is not acquired', self.name)
                return True
            elif self.suppress_inner_lock_ex:
                print('Lock is not acquired in inner context', self.name)
                return True

            print('Lock is not acquired, raising', self.name)

    def close(self):
        try:
            self.lock.release()
        except redis_lock.NotAcquired:
            pass


########################################################################################################################

DEFAULT_WORKER_COUNT = multiprocessing.cpu_count()


class pool_ctx(ContextDecorator):
    """
    Multiprocessing Pool context decorator
    Usage:

        @pool_ctx()
        def some_prog(_ctx, arguments):
            _ctx.pool.apply_async(....)

        @pool_ctx(count=120) # 120 workers
        def some_prog(_ctx, arguments):
            _ctx.pool.apply_async(....)

        @pool_ctx(multiplier=12, logger=log) # 12*cpu_count workers
        def some_prog(_ctx, arguments):
            _ctx.pool.apply_async(....)

    """

    def __init__(self, count=DEFAULT_WORKER_COUNT, multiplier=1, logger=default_logger):
        self.logger = logger
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.count = count * multiplier
        self.pool = multiprocessing.Pool(self.count)
        signal.signal(signal.SIGINT, original_sigint_handler)
        self.push_context = True

    def __enter__(self):
        self.logger.info('Initializing {count} workers'.format(count=self.count))
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if exc_type == KeyboardInterrupt:
            self.logger.info('Caught KeyboardInterrupt, terminating workers')
            self.pool.terminate()
            return True
        elif not exc_type:
            self.logger.info('Normal termination, pool closed')
            self.pool.close()
            return True


########################################################################################################################

if __name__ == '__main__':

    with track_exec_time():
        print 2 ** 2

    TYPE_CHECK = STRONG


    @pool_ctx(count=120)
    def ff(_ctx, x):
        print x ** x


    @pool_ctx()
    def fact(_ctx, x):

        n = 1
        xx = 1
        while n < x:
            xx *= n
            n += 1
            # print n

        try:
            ff(12)
        except:
            print 3334
        # print 111
        time.sleep(2)
        ff(12)
        ff(12)

        return xx


    ff(10)
    print fact(10)
