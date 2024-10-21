import { useAppSelector } from '@lib/hooks/redux';
import AccidentAlarm from '~/components/Layout/AccidentAlarm/AccidentAlarm';
import AdminHome from './AdminHome';
import ClientHome from './ClientHome';
import DefaultHome from './DefaultHome';

const Home = () => {
  const member = useAppSelector((state) => state.member).value;
  return (
    <>
      {member.isAuth ? (
        member.role === 'ROLE_SUPER_ADMIN' ? (
          <AdminHome />
        ) : (
          <>
            <ClientHome />
            <AccidentAlarm />
          </>
        )
      ) : (
        <DefaultHome />
      )}
    </>
  );
};

export default Home;
