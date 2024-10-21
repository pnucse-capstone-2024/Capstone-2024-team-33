import {
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
} from '@mui/material';
import { BarChart } from '@mui/x-charts';
import { DatasetType } from '@mui/x-charts/internals';
import { Dispatch, SetStateAction } from 'react';
import {
  GroupByType,
  PeriodType,
  StatisticType,
  barChartSeriesByReason,
  barChartSeriesByTotal,
  periodDataKey,
} from '~/data/statistic';

interface StatisticBarChartProps {
  statisticType: StatisticType;
  setStatisticType: Dispatch<SetStateAction<StatisticType>>;
  years: number[];
  datasetByYear?: DatasetType;
  datasetByMonth?: DatasetType;
  isLoading: boolean;
}

const StatisticBarChart = ({
  statisticType,
  setStatisticType,
  years,
  datasetByYear,
  datasetByMonth,
  isLoading,
}: StatisticBarChartProps) => {
  return (
    <div className="w-full min-w-[380px]">
      {isLoading ? (
        <div className="flex items-center justify-center">
          <CircularProgress />
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          <div className="flex gap-2">
            <FormControl fullWidth className="max-w-[110px]">
              <InputLabel id="period-select-label">기간</InputLabel>
              <Select
                labelId="period-select-label"
                id="peiod-select"
                value={statisticType.period}
                label="기간"
                size="small"
                onChange={(e) => {
                  const value = e.target.value as PeriodType;
                  if (value === PeriodType.MONTH)
                    setStatisticType((prev) => {
                      return {
                        ...prev,
                        period: e.target.value as PeriodType,
                        year: years?.[0].toString() ?? '-',
                      };
                    });
                  else
                    setStatisticType((prev) => {
                      return {
                        ...prev,
                        period: e.target.value as PeriodType,
                        year: '-',
                      };
                    });
                }}
              >
                {Object.values(PeriodType).map((periodType) => (
                  <MenuItem key={periodType} value={periodType}>
                    {periodType}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth className="max-w-[110px]">
              <InputLabel id="year-select-label">연도</InputLabel>
              <Select
                labelId="year-select-label"
                id="year-select"
                value={statisticType.year}
                label="연도"
                size="small"
                onChange={(e) =>
                  setStatisticType((prev) => {
                    return {
                      ...prev,
                      year: e.target.value,
                    };
                  })
                }
                disabled={statisticType.period !== PeriodType.MONTH}
              >
                {statisticType.period !== PeriodType.MONTH ? (
                  <MenuItem value="-">-</MenuItem>
                ) : (
                  years?.map((year) => (
                    <MenuItem key={year} value={year}>
                      {year}
                    </MenuItem>
                  ))
                )}
              </Select>
            </FormControl>
          </div>

          <FormControl fullWidth className="max-w-[110px]">
            <InputLabel id="groupby-select-label">그룹화 기준</InputLabel>
            <Select
              labelId="groupby-select-label"
              id="peiod-select"
              value={statisticType.groupby}
              label="그룹화 기준"
              size="small"
              onChange={(e) =>
                setStatisticType((prev) => {
                  return {
                    ...prev,
                    groupby: e.target.value as GroupByType,
                  };
                })
              }
            >
              {Object.values(GroupByType).map((groupbyType) => (
                <MenuItem key={groupbyType} value={groupbyType}>
                  {groupbyType}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <BarChart
            xAxis={[
              {
                dataKey: periodDataKey[statisticType.period],
                scaleType: 'band',
              },
            ]}
            yAxis={[{ label: '건 수(건)' }]}
            series={
              statisticType.groupby === GroupByType.REASON
                ? barChartSeriesByReason
                : barChartSeriesByTotal
            }
            dataset={
              (statisticType.period === PeriodType.YEAR
                ? datasetByYear
                : datasetByMonth) ?? [{}]
            }
            height={300}
          />
        </div>
      )}
    </div>
  );
};

export default StatisticBarChart;
