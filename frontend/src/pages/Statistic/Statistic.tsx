import ContentBox from '~/components/ContentBox';
import GlobalAccidentInformations from '~/components/Statistic/GlobalAccidentInformations';
import GlobalPerformanceStatistics from '~/components/Statistic/GlobalPerformanceStatistics';
import GlobalStatisticBarChart from '~/components/Statistic/GlobalStaticBarChart';
import PageCenterTitle from '~/components/Typography/PageCenterTitle';

const Statistic = () => {
  return (
    <>
      <PageCenterTitle title="국내 낙상사고 통계" />
      <ContentBox title="사고 통계">
        <div className="flex flex-col gap-8 lg:flex-row">
          <GlobalStatisticBarChart />
          <GlobalPerformanceStatistics />
        </div>
      </ContentBox>
      <ContentBox title="사고 이력">
        <GlobalAccidentInformations />
      </ContentBox>
    </>
  );
};

export default Statistic;
