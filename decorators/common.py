import functools
import multiprocessing
import signal
import sys
import warnings


########################################################################################################################

class ContextDecorator(object):
    push_context = False
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

