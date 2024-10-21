import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import cookie from 'react-cookies';
import { Role } from '~/types/common.dto';

export interface AuthState {
  name: string;
  memberId: number;
  hospitalId: number[];
  role: Role;
  token: string;
}

export interface AuthSliceState extends AuthState {
  isAuth: boolean; // 로그인되어 있는지.
}

interface InitialState {
  value: AuthSliceState;
}

const initialState: InitialState = {
  value: {
    name: '',
    memberId: -1,
    hospitalId: [],
    role: 'ROLE_ANONYMOUS',
    token: '',
    isAuth: false,
  },
};

export const memberSlice = createSlice({
  name: 'member',
  initialState: initialState,
  reducers: {
    login: (state, action: PayloadAction<AuthState>) => {
      cookie.save(
        'auth',
        JSON.stringify({ ...action.payload, isAuth: true }),
        {},
      );
      state.value = { ...action.payload, isAuth: true };
    },
    logout: (state) => {
      cookie.remove('auth');
      state.value = initialState.value;
    },
  },
});

export const { login, logout } = memberSlice.actions;

export default memberSlice.reducer;
