import { DatasetType } from '@mui/x-charts/internals';
import { QueryKeys } from '~/data/queryKey';
import {
  AccidentPatchDto,
  CameraDto,
  GlobalAccidentInformationPageableDto,
  GlobalPerformanceStatisticDto,
  HospitalAccidentInformationPageableDto,
  HospitalDto,
  HospitalPerformanceStatisticDto,
  HospitalStatisticDto,
  LoginDto,
  MemberCreateDto,
  MonitorDto,
  MonitorPatchDto,
  MyInformationDto,
  UnprocessedAccidentInformationDto,
} from '~/types/common.dto';
import { authClient, client } from '../api/client.axios';
import { useAxiosMutation, useAxiosQuery } from './useAxios';

export const useLogin = () =>
  useAxiosMutation({
    mutationFn: async ({
      loginId,
      loginPassword,
    }: {
      loginId: string;
      loginPassword: string;
    }): Promise<LoginDto | null> => {
      const response = await client.post(`/auth/login`, {
        loginId,
        loginPassword,
      });
      return response?.data;
    },
  });

export const useLogout = () =>
  useAxiosMutation({
    mutationFn: async () => {
      await client.post(`/auth/logout`);
    },
  });

export const useExistMemberIdQuery = () =>
  useAxiosMutation({
    mutationFn: async (loginId: string) => {
      await client.get(`/auth/login-id/${loginId}`);
    },
  });

export const useHospitalSearchQuery = () =>
  useAxiosMutation({
    mutationFn: async (keyword: string): Promise<HospitalDto[]> => {
      const response = await client.get(`/hospitals`, { params: { keyword } });
      return response?.data;
    },
  });

export const useMemberCreate = () =>
  useAxiosMutation({
    mutationFn: async ({
      loginId,
      loginPassword,
      adminName,
      adminTel,
      hospitalId,
    }: MemberCreateDto) => {
      await client.post(`/auth/signup`, {
        loginId,
        loginPassword,
        adminName,
        adminTel,
        hospitalId,
      });
    },
  });
export const useMyInformationQuery = () =>
  useAxiosQuery({
    queryKey: QueryKeys.MY_INFORMATION,
    queryFn: async (): Promise<MyInformationDto | null> => {
      const response = await authClient.get(`/auth/members/me`);
      return response?.data;
    },
  });

export const useHospitalStatisticQuery = (hospitalId: number) =>
  useAxiosQuery({
    queryKey: QueryKeys.HOSPITAL_STATISTIC(hospitalId),
    queryFn: async (): Promise<HospitalStatisticDto | null> => {
      const response = await client.get(`/hospitals/${hospitalId}`);
      return response?.data;
    },
  });

// 막대 통계 그래프 데이터
export const useGlobalStatisticByYearQuery = (options?: {
  enabled?: boolean;
}) =>
  useAxiosQuery({
    ...options,
    queryKey: QueryKeys.GLOBAL_STATISTIC_BY_YEAR,
    queryFn: async (): Promise<DatasetType> => {
      const response = await client.get(`/statistics/year`);
      return response?.data;
    },
  });

export const useGlobalStatisticByMonthQuery = (
  year: string,
  options?: {
    enabled?: boolean;
  },
) =>
  useAxiosQuery({
    ...options,
    queryKey: QueryKeys.GLOBAL_STATISTIC_BY_MONTH(year),
    queryFn: async (): Promise<DatasetType> => {
      const response = await client.get(`/statistics/month`, {
        params: { year },
      });
      return response?.data;
    },
  });

export const useGlobalStatisticYearQuery = () =>
  useAxiosQuery({
    queryKey: QueryKeys.GLOBAL_STATISTIC_YEAR,
    queryFn: async (): Promise<number[]> => {
      const response = await client.get(`/statistics/exist/years`);
      return response?.data;
    },
  });

export const useHospitalStatisticByYearQuery = (
  hospitalId: number,
  options?: { enabled?: boolean },
) =>
  useAxiosQuery({
    ...options,
    queryKey: QueryKeys.HOSPITAL_STATISTIC_BY_YEAR(hospitalId),
    queryFn: async (): Promise<DatasetType> => {
      const response = await authClient.get(
        `/hospitals/${hospitalId}/statistics/year`,
      );
      return response?.data;
    },
  });

export const useHospitalStatisticByMonthQuery = (
  year: string,
  hospitalId: number,
  options?: {
    enabled?: boolean;
  },
) =>
  useAxiosQuery({
    ...options,
    queryKey: QueryKeys.HOSPITAL_STATISTIC_BY_MONTH(year, hospitalId),
    queryFn: async (): Promise<DatasetType> => {
      const response = await authClient.get(
        `/hospitals/${hospitalId}/statistics/month`,
        {
          params: { year },
        },
      );
      return response?.data;
    },
  });
export const useHospitalStatisticYearQuery = (hospitalId: number) =>
  useAxiosQuery({
    queryKey: QueryKeys.HOSPITAL_STATISTIC_YEAR(hospitalId),
    queryFn: async (): Promise<number[]> => {
      const response = await authClient.get(
        `/statistics/exist/years/hospital/${hospitalId}`,
      );
      return response?.data;
    },
  });

// 통계 성과 조회
export const useGlobalPerformanceStatisticQuery = () =>
  useAxiosQuery({
    queryKey: QueryKeys.GLOBAL_PERFORMANCE_STATISTIC,
    queryFn: async (): Promise<GlobalPerformanceStatisticDto> => {
      const response = await client.get(`/statistics/performance`);
      return response?.data;
    },
  });

export const useHospitalPerformanceStatisticQuery = (hospitalId: number) =>
  useAxiosQuery({
    queryKey: QueryKeys.HOSPITAL_PERFORMANCE_STATISTIC(hospitalId),
    queryFn: async (): Promise<HospitalPerformanceStatisticDto> => {
      const response = await authClient.get(
        `/statistics/performance/hospitals/${hospitalId}`,
      );
      return response?.data;
    },
  });

// 사고 목록 조회
export const useGlobalAccidentInformationsQuery = (
  pageNumber: number,
  pageSize: number,
) =>
  useAxiosQuery({
    queryKey: QueryKeys.GLOBAL_ACCIDENT_INFORMATIONS(pageNumber, pageSize),
    queryFn: async (): Promise<GlobalAccidentInformationPageableDto> => {
      const response = await client.get(`/accidents`, {
        params: { pageNumber, pageSize },
      });
      return response?.data;
    },
  });

export const useHospitalAccidentInformationsQuery = (
  pageNumber: number,
  pageSize: number,
  hospitalId: number,
) =>
  useAxiosQuery({
    queryKey: QueryKeys.HOSPITAL_ACCIDENT_INFORMATIONS(
      pageNumber,
      pageSize,
      hospitalId,
    ),
    queryFn: async (): Promise<HospitalAccidentInformationPageableDto> => {
      const response = await authClient.get(
        `/accidents/hospitals/${hospitalId}`,
        {
          params: { pageNumber, pageSize },
        },
      );
      return response?.data;
    },
  });

export const useCamerasQuery = (hospitalId: number) =>
  useAxiosQuery({
    queryKey: QueryKeys.CAMERAS(hospitalId),
    queryFn: async (): Promise<CameraDto[]> => {
      const response = await authClient.get(`/hospitals/${hospitalId}/cameras`);
      return response?.data;
    },
  });

export const useMonitorsQuery = (hospitalId: number) =>
  useAxiosQuery({
    queryKey: QueryKeys.MONITORS(hospitalId),
    queryFn: async (): Promise<(MonitorDto | null)[]> => {
      const response = await authClient.get(
        `/hospitals/${hospitalId}/monitors`,
      );
      return response?.data;
    },
  });

export const useMonitorChangePatch = () =>
  useAxiosMutation({
    mutationFn: async ({ hospitalId, slot, cameraId }: MonitorPatchDto) => {
      await authClient.patch(`/hospitals/${hospitalId}/monitors`, {
        slot,
        cameraId,
      });
    },
  });

export const useAccidentTypeQuery = () =>
  useAxiosQuery({
    queryKey: QueryKeys.ACCIDENT_TYPES,
    queryFn: async (): Promise<{ id: number; content: string }[]> => {
      const response = await authClient.get(`/accidents/types`);
      return response?.data;
    },
  });
export const useAccidentAgeQuery = () =>
  useAxiosQuery({
    queryKey: QueryKeys.ACCIDENT_AGES,
    queryFn: async (): Promise<{ id: number; content: string }[]> => {
      const response = await authClient.get(`/accidents/ages`);
      return response?.data;
    },
  });

export const useAccidentChangePatch = () =>
  useAxiosMutation({
    mutationFn: async ({ accidentId, type, age }: AccidentPatchDto) => {
      await authClient.patch(`/accidents/${accidentId}`, {
        type,
        age,
      });
    },
  });

export const useUnprocessedAccidentQuery = (hospitalId: number) =>
  useAxiosQuery({
    queryKey: QueryKeys.UNPROCESSED_ACCIDENT(hospitalId),
    queryFn: async (): Promise<UnprocessedAccidentInformationDto[]> => {
      const response = await authClient.get(
        `/accidents/hospitals/${hospitalId}/unprocessed`,
      );
      return response?.data;
    },
  });
