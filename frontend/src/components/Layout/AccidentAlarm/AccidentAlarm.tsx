import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useRef, useState } from 'react';
import { MdWarning } from 'react-icons/md';
import { toast } from 'react-toastify';
import { QueryKeys } from '~/data/queryKey';
import { useAppSelector } from '~/lib/hooks/redux';
import { useUnprocessedAccidentQuery } from '~/lib/hooks/useApi';
import {
  useAccidentAlarmByMike,
  useAccidentAlarmByVideo,
} from '~/lib/hooks/useStompApi';
import { UnprocessedAccidentInformationDto } from '~/types/common.dto';
import AccidentRegisterModal from '../../Modal/AccidentRegisterModal';

const AccidentAlarm = () => {
  const hospitalId = useAppSelector((state) => state.member).value
    .hospitalId?.[0];
  const [selectedAccident, setSelectedAccident] =
    useState<UnprocessedAccidentInformationDto | null>(null);
  const { data: unprocessedAccidents } =
    useUnprocessedAccidentQuery(hospitalId);
  const queryClient = useQueryClient();
  const accidentsRef = useRef<UnprocessedAccidentInformationDto[] | undefined>(
    unprocessedAccidents,
  );

  const handleAccidentAlarm = useCallback(async () => {
    queryClient.invalidateQueries({
      queryKey: QueryKeys.UNPROCESSED_ACCIDENT(hospitalId),
    });
  }, []);
  useAccidentAlarmByVideo(hospitalId, handleAccidentAlarm);
  useAccidentAlarmByMike(hospitalId, handleAccidentAlarm);

  useEffect(() => {
    if (unprocessedAccidents) {
      accidentsRef.current = unprocessedAccidents;
      unprocessedAccidents.forEach((accident) =>
        toast(
          () => (
            <div
              className={`flex rounded-sm border-2 ${accident.camera ? 'border-red-500' : 'border-yellow-500'} bg-white transition-shadow hover:shadow-md`}
              onClick={() => setSelectedAccident(accident)}
            >
              <div
                className={`flex items-center justify-center ${accident.camera ? 'bg-red-500' : 'bg-yellow-500'} p-1 text-white`}
              >
                <MdWarning className="h-6 w-6" />
              </div>
              <div className="flex flex-col gap-2 p-4 text-sm">
                <p>
                  <span className="font-bold text-primary">
                    {accident.camera?.content || accident.mike?.content}{' '}
                    {accident.camera ? '카메라' : '마이크'}
                  </span>
                  에서 사고가 감지되었습니다.
                </p>
                <p className="text-xs text-comment">
                  ※ 클릭하면 사고 대응을 위한 모달이 열립니다.
                </p>
              </div>
            </div>
          ),
          {
            className: 'bg-transparent w-fit shadow-none overflow-visible p-0',
            bodyClassName: 'w-fit overflow-visible p-0',
            position: 'top-left',
            autoClose: false,
            hideProgressBar: true,
            closeButton: false,
            draggable: false,
            toastId: accident.id,
          },
        ),
      );
    }
    return () => {
      if (accidentsRef.current) {
        accidentsRef.current.forEach((accident) => toast.dismiss(accident.id));
        console.log(accidentsRef.current);
      }
    };
  }, [unprocessedAccidents]);
  //   useEffect(() => {
  //     return () => {
  //       if (accidentsRef.current) {
  //         accidentsRef.current.forEach((accident) => toast.dismiss(accident.id));
  //         console.log(accidentsRef.current);
  //       }
  //     };
  //   }, []);

  return (
    <AccidentRegisterModal
      selectedAccident={selectedAccident}
      onClose={() => setSelectedAccident(null)}
    />
  );
};

export default AccidentAlarm;
