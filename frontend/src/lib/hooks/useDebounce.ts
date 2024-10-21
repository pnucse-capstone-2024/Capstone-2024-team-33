import { useCallback, useEffect } from 'react';

function useDebounce<T>(func: () => void, delay: number, deps: T) {
  const callback = useCallback(func, [deps]);

  useEffect(() => {
    const timer = setTimeout(() => {
      callback();
    }, delay);

    return () => {
      clearTimeout(timer);
    };
  }, [callback, delay]);
}

export default useDebounce;
