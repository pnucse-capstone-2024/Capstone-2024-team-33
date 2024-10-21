import axios from 'axios';

import cookies from 'react-cookies';
import { categorizeError } from '~/types/error';

export const client = axios.create({
  baseURL: process.env.REACT_APP_SERVER_URL,
  withCredentials: true,
});

client.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(categorizeError(error)),
);

export const authClient = axios.create({
  baseURL: process.env.REACT_APP_SERVER_URL,
  withCredentials: true,
});

authClient.interceptors.request.use((config) => {
  const auth = cookies.load('auth');
  if (auth?.token) {
    config.headers.Authorization = `Bearer ${auth.token}`;
  }
  return config;
});

authClient.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(categorizeError(error)),
);
