import InfraredCameraImage from '@assets/images/infrared_camera_image.png';
import StatisticGraph from '@assets/images/statistic_graph.jpeg';
import HomeSection from '@components/DefaultHome/HomeSection';

const DefaultHome = () => {
  return (
    <div className="flex flex-col gap-10">
      <div className="text-center text-3xl">
        <p>우리 병실의 안전사고 지킴이,</p>
        <p className="font-bold">Safe Catcher</p>
      </div>
      <HomeSection
        title="실시간 스트리밍과 사고 영상 수집"
        content={
          <>
            Safe Catcher는 병실 내에서 발생하는 낙상사고를 감지하고,
            <br /> 사고 상황을 녹화하여 영상으로 제공해요.
            <br /> 또한, 실시간 스트리밍을 통해 병실 상황을 모니터링할 수
            있어요.
            <br /> 이 기능을 이용하기 위해서는 병원 계정으로 <b>로그인</b>이
            필요해요.
          </>
        }
        buttonText="로그인하러 가기"
        buttonUrl="/login"
        imageSource={InfraredCameraImage}
        imagePosition="right"
      />
      <HomeSection
        title="전국적인 사고 현황 통계"
        content={
          <>
            Safe Catcher는 전국의 노인복지시설로부터
            <br />
            낙상사고가 발생한 기록을 수집해요.
            <br />
            아래에서 국내 낙상 사고 통계를 한 눈에 확인할 수 있어요.
          </>
        }
        buttonText="확인하러 가기"
        buttonUrl="/statistic"
        imageSource={StatisticGraph}
        imagePosition="left"
      />
    </div>
  );
};

export default DefaultHome;
