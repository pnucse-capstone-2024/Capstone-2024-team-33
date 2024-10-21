/* eslint-disable @typescript-eslint/no-explicit-any */
import { Client } from '@stomp/stompjs';
import { useEffect } from 'react';
import SockJS from 'sockjs-client';
import { AppDispatch } from '~/store';
import { connectStomp, disconnectStomp } from '~/store/stompClient.slice';
import { useAppDispatch } from '../hooks/redux';

type Props = {
  children: React.ReactNode;
};

const StompClientProvider = ({ children }: Props) => {
  const dispatch = useAppDispatch();

  useEffect(() => {
    dispatch(initializeStompClient());

    return () => {
      dispatch(disconnectStomp());
    };
  }, [dispatch]);

  const initializeStompClient = () => (dispatch: AppDispatch) => {
    const stompClient = new Client({
      webSocketFactory: () => new SockJS(`${process.env.REACT_APP_WEB_SOCKET}`),
      onConnect: () => {
        dispatch(connectStomp(stompClient));
      },
      onStompError: (frame: any) => {
        console.error('Broker reported error: ' + frame.headers['message']);
        console.error('Additional details: ' + frame.body);
      },
      onDisconnect: () => {
        dispatch(disconnectStomp());
      },
      onWebSocketClose: () => {
        dispatch(disconnectStomp());
      },
    });

    stompClient.activate();
  };

  return <>{children}</>;
};
export default StompClientProvider;
