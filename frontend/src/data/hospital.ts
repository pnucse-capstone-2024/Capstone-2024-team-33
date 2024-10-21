import { CameraDto } from '~/types/common.dto';

export type HospitalType =
  | 'HOSPITAL'
  | 'OLD_NURSING_HOSPITAL'
  | 'NURSING_HOSPITAL'
  | 'DISABLED_NURSING_HOSPITAL'
  | 'MENTAL_HOSPITAL'
  | 'GENERAL_HOSPITAL'
  | 'DENTISTRY_HOSPITAL'
  | 'ORIENTAL_MEDICINE_HOSPITAL';

export const hospitalTypeLabel = {
  HOSPITAL: '병원',
  OLD_NURSING_HOSPITAL: '요양병원(노인병원)',
  NURSING_HOSPITAL: '요양병원(일반요양병원)',
  DISABLED_NURSING_HOSPITAL: '요양병원(장애인의료재활시설)',
  MENTAL_HOSPITAL: '정신병원',
  GENERAL_HOSPITAL: '종합병원',
  DENTISTRY_HOSPITAL: '치과병원',
  ORIENTAL_MEDICINE_HOSPITAL: '한방벙원',
};

export const mockCameras: CameraDto[] = [
  { id: 1, content: '401호(일반 병실)' },
  { id: 2, content: '402호(일반 병실)' },
  { id: 3, content: '403호(일반 병실)' },
  { id: 4, content: '404호(일반 병실)' },
  { id: 5, content: '405호(일반 병실)' },
];

export const mockMonitors = [
  { id: 1, content: '401호(일반 병실)' },
  { id: 3, content: '403호(일반 병실)' },
  { id: 2, content: '402호(일반 병실)' },
  { id: 4, content: '404호(일반 병실)' },
  null,
];
