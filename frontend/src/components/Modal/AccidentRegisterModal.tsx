import {
  Button,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
} from '@mui/material';
import { useEffect, useRef, useState } from 'react';
import { toast } from 'react-toastify';
import {
  useAccidentAgeQuery,
  useAccidentChangePatch,
  useAccidentTypeQuery,
} from '~/lib/hooks/useApi';
import useBodyScrollLock from '~/lib/hooks/useBodyScrollLock';
import useClickOutside from '~/lib/hooks/useOnClickOutside';
import { formatDateTime } from '~/lib/utils/util';
import { UnprocessedAccidentInformationDto } from '~/types/common.dto';

interface AccidentRegisterModalProps {
  selectedAccident: UnprocessedAccidentInformationDto | null;
  onClose: () => void;
}

const AccidentRegisterModal = ({
  selectedAccident,
  onClose,
}: AccidentRegisterModalProps) => {
  const ref = useRef<HTMLDivElement>(null);
  const [isSelectOpen, setIsSelectOpen] = useState(false); // Select가 열렸는지 여부를 관리하는 상태
  const [accidentInfo, setAccidentInfo] = useState<{
    type: number;
    age: number;
  }>({
    type: -1,
    age: -1,
  });
  const { lockScroll, unlockScroll } = useBodyScrollLock();
  useClickOutside(ref, () => {
    if (!isSelectOpen) onClose();
  });
  const { data: accidentTypes } = useAccidentTypeQuery();
  const { data: accidentAges } = useAccidentAgeQuery();
  const { mutate } = useAccidentChangePatch();

  useEffect(() => {
    if (selectedAccident) {
      lockScroll();
    } else {
      unlockScroll();
    }

    return () => unlockScroll();
  }, [selectedAccident, lockScroll, unlockScroll]);

  if (!selectedAccident) return <></>;
  return (
    <div className="fixed inset-0 z-[51] flex items-center justify-center bg-black bg-opacity-20">
      <div
        ref={ref}
        className="flex h-[606px] w-[1084px] flex-col rounded-[4px] bg-white"
      >
        <div className="flex justify-center rounded-t-[4px] bg-background py-4">
          <p className="text-lg font-bold">
            {selectedAccident.camera?.content} 사고영상 -{' '}
            {formatDateTime(selectedAccident.startDate)}
          </p>
        </div>
        <div className="flex h-[554px] w-full flex-grow">
          <div className="flex h-full flex-grow justify-center rounded-bl-[4px] bg-black">
            <video autoPlay loop controls className="object-contain">
              <source
                src={`${process.env.REACT_APP_VIDEO_PREFIX}${selectedAccident.videoUrl}`}
                type="video/mp4"
              />
            </video>
          </div>
          <div className="flex w-[236px] flex-col justify-center gap-5 px-2 py-10">
            <FormControl fullWidth>
              <InputLabel id="accident-type-label">사고 유형</InputLabel>
              <Select
                labelId="accident-type-label"
                id="accident-type"
                value={accidentInfo.type}
                label="사고 유형"
                size="small"
                onOpen={() => setIsSelectOpen(true)} // Select가 열렸을 때
                onClose={() => setIsSelectOpen(false)} // Select가 닫혔을 때
                onChange={(e) =>
                  setAccidentInfo((prev) => {
                    return {
                      ...prev,
                      type: e.target.value as number,
                    };
                  })
                }
              >
                {accidentTypes?.map((type) => (
                  <MenuItem key={type.id} value={type.id}>
                    {type.content}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel id="accident-age-label">피해자 연령대</InputLabel>
              <Select
                labelId="accident-age-label"
                id="accident-age"
                value={accidentInfo.age}
                label="피해자 연령대"
                size="small"
                onOpen={() => setIsSelectOpen(true)} // Select가 열렸을 때
                onClose={() => setIsSelectOpen(false)} // Select가 닫혔을 때
                onChange={(e) =>
                  setAccidentInfo((prev) => {
                    return {
                      ...prev,
                      age: e.target.value as number,
                    };
                  })
                }
              >
                {accidentAges?.map((age) => (
                  <MenuItem key={age.id} value={age.id}>
                    {age.content}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <div className="flex justify-center gap-5">
              <Button
                onClick={() =>
                  mutate(
                    {
                      accidentId: selectedAccident.id,
                      type: accidentInfo.type,
                      age: accidentInfo.age,
                    },
                    {
                      onSuccess: () => {
                        toast.success('사고 정보가 성공적으로 등록되었습니다.');
                        toast.dismiss(selectedAccident.id);
                        setAccidentInfo({ type: -1, age: -1 });
                        onClose();
                      },
                      onError: () => {
                        toast.error('사고 정보 등록에 실패하였습니다.');
                      },
                    },
                  )
                }
                variant="contained"
                color="primary"
                sx={{ color: 'white' }}
              >
                등록
              </Button>
              <Button onClick={onClose} variant="contained" color="secondary">
                창 닫기
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AccidentRegisterModal;
