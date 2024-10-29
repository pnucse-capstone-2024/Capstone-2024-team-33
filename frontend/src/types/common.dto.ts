import { HospitalType } from '../data/hospital';

interface PageSort {
  empty: boolean;
  sorted: boolean;
  unsorted: boolean;
}

interface Pageable {
  totalPages: number;
  totalElements: number;
  size: number;
  number: number;
  sort: PageSort;
  first: boolean;
  last: boolean;
  pageable: string;
  numberOfElements: number;
  empty: boolean;
}

export type Role = 'ROLE_ADMIN' | 'ROLE_SUPER_ADMIN' | 'ROLE_ANONYMOUS';

export interface LoginDto {
  id: number;
  hospitalId: number[];
  name: string;
  memberRole: Role;
  accessToken: string;
}

export interface MemberInfo {
  memberId: number;
  loginId: number;
  emailAddress: string;
  realName: string;
  thumbnailPath: string | null;
  generation: string;
  memberRoles: Role[];
}

export interface ExistMemberId {
  isExistMemberId: boolean;
}

export interface MemberCreateDto {
  loginId: string;
  loginPassword: string;
  adminName: string;
  adminTel: string;
  hospitalId: number;
}

export interface HospitalDto {
  id: number;
  name?: string;
  type?: HospitalType;
  address?: string;
  tel: string;
}

export interface MyInformationDto {
  id: number; // member id
  loginId: string;
  name: string;
  memberRole: Role;
  tel: string;
  hospital: {
    id: number;
    name: string;
    tel: string;
    type: HospitalType;
    address: string;
  };
}

export interface HospitalStatisticDto {
  id: number;
  name: string;
  type: HospitalType;
  today: number;
  year: number;
  detectionRate: number;
  accidents: { date: string; accidentCount: number }[];
}

export interface GlobalPerformanceStatisticDto {
  totalAccidentCount: number;
  primaryReason: {
    id: number;
    content: string;
  };
  increaseRateByLastYear: number; // 퍼센트
  detectionAccuracy: number; //퍼센트
  mostAccidentsOrccuredMonth: number;
  mostAccidentsOrccuredYear: number;
}

export interface HospitalPerformanceStatisticDto {
  totalAccidentCount: number;
  primaryReason: {
    id: number;
    content: string;
  };
  increaseRateByLastYear: number; // 퍼센트
  detectionAccuracy: number; //퍼센트
  mostAccidentsOrccuredMonth: number;
  increaseRateByAverage: number; //퍼센트
}

export interface GlobalAccidentInformationDto {
  id: number;
  startTime: string;
  type: {
    id: number;
    content: string;
  };
  age: {
    id: number;
    content: string;
  };
}

export interface GlobalAccidentInformationPageableDto extends Pageable {
  content: GlobalAccidentInformationDto[];
}

export interface HospitalAccidentInformationDto {
  id: number;
  startTime: string;
  type: {
    id: number;
    content: string;
  };
  age: {
    id: number;
    content: string;
  };
  camera: {
    id: number;
    content: string;
  } | null;
  mike: {
    id: number;
    content: string;
  } | null;
  videoUrl: string;
}

export interface HospitalAccidentInformationPageableDto extends Pageable {
  content: HospitalAccidentInformationDto[];
}

export interface UnprocessedAccidentInformationDto {
  id: number;
  startDate: string;
  camera: {
    id: number;
    content: string;
  } | null;
  mike: {
    id: number;
    content: string;
  } | null;
  videoUrl: string;
}

export interface CameraDto {
  id: number;
  content: string;
}

export interface MonitorPatchDto {
  hospitalId: number;
  slot: number; // 0~4
  cameraId: number;
}

export interface AccidentPatchDto {
  accidentId: number;
  type: number;
  age: number;
}

export interface MonitorDto {
  id: number;
  content: string;
}

export interface ImageMessageDto {
  encodedImage: string;
}

export interface AccidentAlarmMessageDto {
  hospitalId: number;
  camera: { id: number; content: string };
  startTime: string;
  endTime: string;
}
