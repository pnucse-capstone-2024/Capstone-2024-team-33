import { useCallback } from 'react';

export default function useBodyScrollLock() {
  const lockScroll = useCallback(() => {
    document.body.style.overflow = 'hidden';
    document.documentElement.style.overflow = 'hidden';
    document.body.style.touchAction = 'none';
  }, []);

  const unlockScroll = useCallback(() => {
    document.body.style.overflow = '';
    document.documentElement.style.overflow = '';
    document.body.style.touchAction = '';
  }, []);

  return { lockScroll, unlockScroll };
}
