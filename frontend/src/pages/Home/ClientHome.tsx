import { Button, CircularProgress, MenuItem, Select } from '@mui/material';
import { IMessage } from '@stomp/stompjs';
import { useCallback, useEffect, useState } from 'react';
import { BiCameraOff } from 'react-icons/bi';
import { MdSettings } from 'react-icons/md';
import { toast } from 'react-toastify';
import ContentBox from '~/components/ContentBox';
import PageCenterTitle from '~/components/Typography/PageCenterTitle';
import { useAppSelector } from '~/lib/hooks/redux';
import {
  useCamerasQuery,
  useMonitorChangePatch,
  useMonitorsQuery,
} from '~/lib/hooks/useApi';
import { useLiveStreaming } from '~/lib/hooks/useStompApi';
import { ImageMessageDto } from '~/types/common.dto';

const ClientHome = () => {
  const hospitalId = useAppSelector((state) => state.member).value
    .hospitalId?.[0];
  const { data: cameras, isLoading: isCameraLoding } =
    useCamerasQuery(hospitalId);
  const {
    data: monitors,
    refetch,
    isLoading: isMonitorLoading,
  } = useMonitorsQuery(hospitalId);
  const { mutate } = useMonitorChangePatch();
  const Monitor = ({
    cameraId,
    slotNumber,
    colspan = 1,
    rowspan = 1,
  }: {
    cameraId: number;
    slotNumber: number;
    colspan?: number;
    rowspan?: number;
  }) => {
    const [imageUrl, setImageUrl] = useState<string>();
    const hospitalId = useAppSelector((state) => state.member).value
      .hospitalId?.[0];

    const handleRecieveImages = useCallback(async (message: IMessage) => {
      const { encodedImage } = JSON.parse(message.body) as ImageMessageDto;

      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
      setImageUrl(encodedImage);
      return () => {
        if (imageUrl) URL.revokeObjectURL(imageUrl);
      };
    }, []);
    useLiveStreaming(hospitalId, cameraId, handleRecieveImages);

    useEffect(() => {
      return () => {
        if (imageUrl) {
          URL.revokeObjectURL(imageUrl);
        }
      };
    }, [imageUrl]);
    return (
      <div
        className={`relative h-full min-h-[150px] rounded-md`}
        style={{
          gridColumn: `span ${colspan} / span ${colspan}`,
          gridRow: `span ${rowspan} / span ${rowspan}`,
        }}
      >
        {imageUrl ? (
          <img
            className="h-full w-full rounded-md object-cover"
            src={`data:image/png;base64,${imageUrl}`}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center rounded-md bg-background">
            <BiCameraOff className="h-8 w-8 text-comment" />
          </div>
        )}
        <div className="absolute bottom-0 left-0 right-0 flex justify-end rounded-b-md bg-black bg-opacity-25 px-2 py-1 text-xs">
          <Select
            size="small"
            variant="standard"
            sx={{
              color: 'white',
              fontSize: '12px',
              padding: '0px',
              '.MuiSelect-select': {
                padding: '1px',
              },
              '&::before': {
                borderBottom: '1px solid white',
              },
              '&:hover:not(.Mui-disabled, .Mui-error)::before': {
                borderBottom: '2px solid white',
              },
            }}
            value={monitors?.[slotNumber]?.id ?? -1}
            onChange={(e) => {
              console.log(cameras);
              mutate(
                {
                  hospitalId,
                  slot: slotNumber,
                  cameraId: e.target.value as number,
                },
                {
                  onSuccess: () => {
                    refetch();
                  },
                  onError: (error) => {
                    toast.error(error.message);
                  },
                },
              );
            }}
          >
            <MenuItem value={-1}>선택 안함</MenuItem>
            {!isCameraLoding &&
              cameras?.map(({ id, content }) => (
                <MenuItem key={id} value={id}>
                  {content}
                </MenuItem>
              ))}
          </Select>
        </div>
      </div>
    );
  };

  return (
    <>
      <PageCenterTitle title="모니터링" />
      <ContentBox>
        <div className="flex justify-end">
          <Button
            color="primary"
            variant="outlined"
            startIcon={<MdSettings />}
            onClick={() => toast.info('아직 준비중인 기능이에요.')}
          >
            카메라 편집
          </Button>
        </div>
        <div className="grid grid-cols-2 grid-rows-3 gap-2 md:grid-cols-4 md:grid-rows-2">
          {isMonitorLoading ? (
            <div className="col-span-2 row-span-4 flex items-center justify-center md:col-span-4 md:row-span-2">
              <CircularProgress />
            </div>
          ) : (
            <>
              <Monitor
                cameraId={monitors?.[0]?.id ?? 0}
                slotNumber={0}
                colspan={2}
                rowspan={2}
              />
              {Array(4)
                .fill(0)
                .map((_, index) => (
                  <Monitor
                    key={index}
                    cameraId={monitors?.[index + 1]?.id ?? 0}
                    slotNumber={index + 1}
                  />
                ))}
            </>
          )}
        </div>
      </ContentBox>
    </>
  );
};

export default ClientHome;
