import { Chip, CircularProgress } from '@mui/material';
import { useAppSelector } from '~/lib/hooks/redux';
import { useHospitalPerformanceStatisticQuery } from '~/lib/hooks/useApi';
import CircleBadge from './CircleBadge';

const HospitalPerformanceStatistics = () => {
  const hospitalId = useAppSelector((state) => state.member).value
    .hospitalId?.[0];
  const { data: performanceStatistic, isLoading } =
    useHospitalPerformanceStatisticQuery(hospitalId);
  const currentYear = new Date().getFullYear();

  return (
    <div className="flex w-full flex-col items-center justify-center gap-4 lg:max-w-[347px]">
      {isLoading ? (
        <div className="flex items-center justify-center">
          <CircularProgress />
        </div>
      ) : (
        <>
          <Chip
            label={`${currentYear} 누적 사고 건 수 ${performanceStatistic?.totalAccidentCount}건`}
            variant="filled"
            color="primary"
            sx={{ color: 'white' }}
          />
          <div className="flex gap-4 lg:flex-col ">
            <div className="flex justify-center gap-4">
              <CircleBadge
                title={`${currentYear - 1}년 대비`}
                content={`${Math.abs(performanceStatistic?.increaseRateByLastYear ?? 0).toFixed(1)}%${(performanceStatistic?.increaseRateByLastYear ?? 0) < 0 ? '감소' : '증가'}`}
              />
              <CircleBadge
                title={`사고 감지 정확도`}
                content={`${performanceStatistic?.detectionAccuracy.toFixed(1)}%`}
              />
            </div>
            <div className="flex justify-center gap-4">
              <CircleBadge
                title={`${currentYear} 주 사고원인`}
                content={performanceStatistic?.primaryReason.content ?? ''}
              />
              <CircleBadge
                title={`최다 사고 발생 월`}
                content={`${performanceStatistic?.mostAccidentsOrccuredMonth}월`}
              />
              <CircleBadge
                title={`평균 사고율 대비`}
                content={`${Math.abs(performanceStatistic?.increaseRateByAverage ?? 0).toFixed(1)}%${(performanceStatistic?.increaseRateByAverage ?? 0) < 0 ? '감소' : '증가'}`}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default HospitalPerformanceStatistics;
