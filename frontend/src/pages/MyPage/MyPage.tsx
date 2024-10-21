import { Button } from '@mui/material';
import ContentBox from '~/components/ContentBox';
import PageCenterTitle from '~/components/Typography/PageCenterTitle';
import { hospitalTypeLabel } from '~/data/hospital';
import { useAppDispatch } from '~/lib/hooks/redux';
import { useMyInformationQuery } from '~/lib/hooks/useApi';
import { formatTel } from '~/lib/utils/util';
import { logout } from '~/store/member.slice';

const MyPage = () => {
  const dispatch = useAppDispatch();
  const { data: myInformation } = useMyInformationQuery();

  return (
    <>
      <PageCenterTitle title="계정 정보 조회" />
      {myInformation && (
        <div className="flex gap-8">
          <ContentBox title="내 병원 정보">
            <div className="flex flex-col items-start gap-4">
              <div className="flex items-center gap-2">
                <p className="text-base font-bold">
                  {myInformation.hospital.name}
                </p>
                <span className="rounded-md bg-comment px-4 py-1 text-xs text-white">
                  {hospitalTypeLabel[myInformation.hospital.type]}
                </span>
              </div>
              <div>
                <p className="mb-2 text-sm">{myInformation.hospital.address}</p>
                <p className="text-sm">Tel: {myInformation.hospital.tel}</p>
              </div>
            </div>
          </ContentBox>
          <ContentBox title="계정 관리자 정보">
            <div className="flex flex-col items-start gap-4">
              <p className="flex text-sm">
                <b className="w-[6em]">ID</b>
                {myInformation.loginId}
              </p>
              <p className="flex text-sm">
                <b className="w-[6em]">관리자 이름</b>
                {myInformation.name}
              </p>
              <p className="flex text-sm">
                <b className="w-[6em]">전화번호</b>
                {formatTel(myInformation.tel)}
              </p>
            </div>
          </ContentBox>
        </div>
      )}
      <ContentBox title="계정 관리" titleCenter>
        <div className="flex justify-center gap-8">
          <Button
            size="large"
            variant="outlined"
            color="error"
            onClick={() => dispatch(logout())}
          >
            로그아웃
          </Button>
          <Button size="large" variant="contained" color="error">
            회원탈퇴
          </Button>
          {/* TODO 탈퇴 로직  */}
        </div>
      </ContentBox>
    </>
  );
};

export default MyPage;
