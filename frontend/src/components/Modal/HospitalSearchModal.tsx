/* eslint-disable @typescript-eslint/no-explicit-any */
import { Button, TextField } from '@mui/material';
import classnames from 'classnames';
import { FormikErrors } from 'formik';
import { ChangeEvent, useEffect, useRef, useState } from 'react';
import { VscClose } from 'react-icons/vsc';
import { hospitalTypeLabel } from '~/data/hospital';
import { useHospitalSearchQuery } from '~/lib/hooks/useApi';
import useBodyScrollLock from '~/lib/hooks/useBodyScrollLock';
import useDebounce from '~/lib/hooks/useDebounce';
import useClickOutside from '~/lib/hooks/useOnClickOutside';
import { Hospital, SignUpInfo } from '~/pages/SignUp/SignUp';
import { HospitalDto } from '~/types/common.dto';
import PageTitle from '../Typography/PageTitle';

interface HospitalSearchModalProps {
  open: boolean;
  setOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setFieldValue: (
    field: string,
    value: any,
    shouldValidate?: boolean | undefined,
  ) => Promise<void | FormikErrors<SignUpInfo>>;
}

const HospitalSearchModal = ({
  open,
  setOpen,
  setFieldValue,
}: HospitalSearchModalProps) => {
  const ref = useRef<HTMLDivElement>(null);
  const [keyword, setKeyword] = useState<string>('');
  const [hospitals, setHospitals] = useState<HospitalDto[]>([]);
  const { mutate: searchHospital, isPending } = useHospitalSearchQuery();
  const { lockScroll, unlockScroll } = useBodyScrollLock();
  useClickOutside(ref, () => setOpen(false));

  useDebounce(
    () => {
      if (keyword) {
        searchHospital(keyword, {
          onSuccess: (data) => {
            setHospitals(data);
          },
        });
      }
    },
    300,
    keyword,
  );

  useEffect(() => {
    if (open) {
      lockScroll();
    } else {
      unlockScroll();
    }

    return () => unlockScroll();
  }, [open, lockScroll, unlockScroll]);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    setKeyword(e.target.value);
  };

  const closeModal = () => {
    setOpen(false);
    setKeyword('');
    setHospitals([]);
  };

  if (!open) {
    return null;
  }
  return (
    <div className="fixed left-0 top-0 z-[51] flex h-[100vh] w-[100vw] items-center justify-center bg-black bg-opacity-30">
      <div
        ref={ref}
        className="flex h-[840px] w-[640px] flex-col gap-8 rounded-lg bg-white p-8"
      >
        <div className="flex justify-between">
          <PageTitle title="병원 찾기" />
          <button type="button" className="text-comment" onClick={closeModal}>
            <VscClose className="h-8 w-8" />
          </button>
        </div>
        <TextField
          name="searchKeyword"
          type="text"
          label="병원 검색어"
          className="w-full"
          value={keyword}
          onChange={handleChange}
        />
        <div
          className={classnames(
            'relative flex min-h-0 w-full flex-auto flex-col rounded-lg border-2 border-gray-300 bg-gray-50 p-4',
            isPending ? 'overflow-hidden' : 'overflow-y-scroll',
          )}
        >
          {hospitals && hospitals.length > 0 ? (
            hospitals.map((hospital) => (
              <div
                key={hospital.id}
                className="flex w-full justify-between gap-4 border-b border-gray-300 p-4"
              >
                <div className="flex flex-col gap-1">
                  <p className="flex flex-wrap items-center gap-x-2">
                    <b>{hospital.name}</b>
                    <span className="rounded-md bg-gray-300 px-2 py-1 text-xs">
                      {hospital.type && hospitalTypeLabel[hospital.type]}
                    </span>
                  </p>
                  <p className="text-sm text-comment">{hospital.address}</p>
                  <p className="text-sm text-comment">TEL: {hospital.tel}</p>
                </div>
                <Button
                  type="button"
                  variant="outlined"
                  color="primary"
                  onClick={() => {
                    setFieldValue('hospital', {
                      id: hospital.id,
                      name: hospital.name,
                      type: hospital.type,
                    } as Hospital);
                    closeModal();
                  }}
                >
                  선택
                </Button>
              </div>
            ))
          ) : (
            <div className="my-auto text-center text-xl text-comment">
              검색된 병원이 없습니다.
            </div>
          )}
          {isPending && (
            <div className="absolute left-0 top-0 flex h-full w-full items-center justify-center bg-white bg-opacity-50 text-lg font-bold text-comment">
              병원을 검색중입니다.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HospitalSearchModal;
