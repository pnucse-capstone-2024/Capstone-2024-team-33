import { messageCallbackType } from '@stomp/stompjs';
import { useSubscribe } from './useStomp';

export function useLiveStreaming(
  hospitalId: number,
  cameraId: number,
  callback: messageCallbackType,
) {
  useSubscribe(
    `/ws/topic/image/hospitals/${hospitalId}/cameras/${cameraId}`,
    callback,
  );
}

export function useAccidentAlarmByVideo(
  hospitalId: number,
  callback: messageCallbackType,
) {
  useSubscribe(`/ws/topic/accident/hospitals/${hospitalId}`, callback);
}

export function useAccidentAlarmByMike(
  hospitalId: number,
  callback: messageCallbackType,
) {
  useSubscribe(`/ws/topic/accident/hospitals/${hospitalId}/audio`, callback);
}
