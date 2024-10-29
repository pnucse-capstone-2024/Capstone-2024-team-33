import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Client } from '@stomp/stompjs';

// 1. 초기 상태 정의
interface StompClientState {
  stompClient: Client | null;
  connected: boolean;
}

const initialState: StompClientState = {
  stompClient: null,
  connected: false,
};

// 2. slice 정의
const stompClientSlice = createSlice({
  name: 'stompClient',
  initialState,
  reducers: {
    connectStomp(state, action: PayloadAction<Client>) {
      state.stompClient = action.payload;
      state.connected = true;
    },
    disconnectStomp(state) {
      state.stompClient = null;
      state.connected = false;
    },
  },
});

export const { connectStomp, disconnectStomp } = stompClientSlice.actions;
export default stompClientSlice.reducer;
