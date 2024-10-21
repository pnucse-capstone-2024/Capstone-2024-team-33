/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */
import createWebStorage from 'redux-persist/lib/storage/createWebStorage';

const createNoopStorage = () => ({
  getItem(_key: any) {
    return Promise.resolve(null);
  },
  setItem(_key: any, value: any) {
    return Promise.resolve(value);
  },
  removeItem(_key: any) {
    return Promise.resolve();
  },
});

const storage =
  typeof window === 'undefined'
    ? createNoopStorage()
    : createWebStorage('local');

export default storage;
