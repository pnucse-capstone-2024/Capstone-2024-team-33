import {
  DefaultError,
  useMutation,
  UseMutationOptions,
  useQuery,
  UseQueryOptions,
} from '@tanstack/react-query';
import { AxiosError } from 'axios';
import { useEffect } from 'react';

import { redirect, useLocation } from 'react-router-dom';
import { toast } from 'react-toastify';
import {
  AccessDeniedError,
  BusinessError,
  UnauthorizedError,
} from '~/types/error';

const handleError = (error: Error, pathname: string) => {
  if (error instanceof UnauthorizedError) {
    redirect(
      `${process.env.CLIENT_URL}/login?redirect=${process.env.CLIENT_URL + pathname}`,
    );
  }

  if (error instanceof AccessDeniedError) {
    toast.error('해당 요청을 수행할 권한이 없습니다.');
    return;
  }

  if (
    error instanceof BusinessError &&
    error.originalError instanceof AxiosError &&
    error.originalError.response
  ) {
    return;
  }

  throw error;
};

export const useAxiosQuery = <TQueryFnData>(
  options: UseQueryOptions<TQueryFnData>,
) => {
  const query = useQuery<TQueryFnData>(options);
  const { error, isError } = query;
  const { pathname } = useLocation();

  useEffect(() => {
    if (isError) {
      handleError(error, pathname);
    }
  }, [error, isError, pathname]);

  return query;
};

export const useAxiosMutation = <TData, TVariables>(
  options: UseMutationOptions<TData, DefaultError, TVariables>,
) => {
  const mutation = useMutation<TData, DefaultError, TVariables>(options);
  const { error, isError } = mutation;
  const { pathname } = useLocation();

  useEffect(() => {
    if (isError) {
      handleError(error, pathname);
    }
  }, [error, isError, pathname]);

  return mutation;
};
