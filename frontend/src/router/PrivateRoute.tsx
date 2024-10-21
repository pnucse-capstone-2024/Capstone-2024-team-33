import { Navigate } from 'react-router-dom';
import { useAppSelector } from '~/lib/hooks/redux';
import { Role } from '~/types/common.dto';

// 로그인 상태를 확인하여 리디렉션을 처리하는 컴포넌트
const PrivateRoute = ({
  element,
  redirectTo,
  blockList,
}: {
  element: JSX.Element;
  redirectTo: string;
  blockList: Role[];
}) => {
  const role =
    useAppSelector((state) => state.member.value.role) ?? 'ROLE_ANONYMOUS';
  return blockList.includes(role) ? <Navigate to={redirectTo} /> : element;
};

export default PrivateRoute;
