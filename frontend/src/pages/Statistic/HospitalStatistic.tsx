import ContentBox from '~/components/ContentBox';
import HospitalAccidentInformations from '~/components/Statistic/HospitalAccidentInformations';
import HospitalPerformanceStatistics from '~/components/Statistic/HospitalPerformanceStatistics';
import HospitalStatisticBarChart from '~/components/Statistic/HospitalStaticBarChart';
import PageCenterTitle from '~/components/Typography/PageCenterTitle';
import { useAppSelector } from '~/lib/hooks/redux';

const HospitalStatistic = () => {
  const member = useAppSelector((state) => state.member).value;

  return (
    <>
      <PageCenterTitle title={`${member.name} 낙상사고 통계`} />
      <ContentBox title="사고 통계">
        <div className="flex flex-col gap-8 lg:flex-row">
          <HospitalStatisticBarChart />
          <HospitalPerformanceStatistics />
        </div>
      </ContentBox>
      <ContentBox title="사고 이력">
        <HospitalAccidentInformations />
      </ContentBox>
    </>
  );
};

export default HospitalStatistic;
