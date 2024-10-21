import { messageCallbackType } from '@stomp/stompjs';
import { useEffect } from 'react';
import { useAppSelector } from './redux';

export function useSubscribe(
  destination: string,
  callback: messageCallbackType,
) {
  const { stompClient, connected } = useAppSelector(
    (state) => state.stompClient,
  );

  useEffect(() => {
    if (!connected || !stompClient) return;
    const subscription = stompClient.subscribe(destination, callback);
    return () => {
      subscription.unsubscribe();
    };
  }, [destination, callback, stompClient, connected]);
}

interface UsePublish {
  publish: (message?: string) => void;
}

export function usePublish(destination: string): UsePublish {
  const { stompClient } = useAppSelector((state) => state.stompClient);

  function publish(message?: string) {
    if (stompClient?.connected) {
      stompClient.publish({ destination, body: message });
    }
  }

  return { publish };
}
