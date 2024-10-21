/* eslint-disable implicit-arrow-linebreak */
import { combineReducers, configureStore } from '@reduxjs/toolkit';
import { persistReducer, persistStore } from 'redux-persist';

import memberSlice from './member.slice';
import stompClientSlice from './stompClient.slice';
import storage from './store';

export const rootReducer = combineReducers({
  member: memberSlice,
  stompClient: stompClientSlice,
});

const persistConfig = {
  key: 'root',
  storage,
  blacklist: ['stompClient'],
};

const persistedReducer = persistReducer(persistConfig, rootReducer);

export const store = configureStore({
  reducer: persistedReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false,
    }),
});
export const persistor = persistStore(store);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
