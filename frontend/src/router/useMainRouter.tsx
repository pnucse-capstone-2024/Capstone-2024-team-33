import Layout from '@components/Layout/Layout';
import Home from '@pages/Home/Home';
import Login from '@pages/Login/Login';
import MyPage from '@pages/MyPage/MyPage';
import NotFound from '@pages/NotFound/NotFound';
import SignUp from '@pages/SignUp/SignUp';
import Statistic from '@pages/Statistic/Statistic';
import { useRoutes } from 'react-router-dom';
import HospitalStatistic from '~/pages/Statistic/HospitalStatistic';
import PrivateRoute from './PrivateRoute';

const useMainRouter = () =>
  useRoutes([
    {
      path: '/',
      element: <Layout />,
      children: [
        { index: true, element: <Home /> },
        { path: '*', element: <NotFound /> },
        {
          path: 'login',
          element: (
            <PrivateRoute
              element={<Login />}
              redirectTo="/"
              blockList={['ROLE_ADMIN', 'ROLE_SUPER_ADMIN']}
            />
          ),
        },
        {
          path: 'statistic',
          element: <Statistic />,
        },
        {
          path: 'sign-up',
          element: (
            <PrivateRoute
              element={<SignUp />}
              redirectTo="/"
              blockList={['ROLE_ADMIN', 'ROLE_SUPER_ADMIN']}
            />
          ),
        },
        {
          path: 'my',
          element: (
            <PrivateRoute
              element={<MyPage />}
              redirectTo="/"
              blockList={['ROLE_ANONYMOUS']}
            />
          ),
        },
        {
          path: 'statistic/hospital',
          element: (
            <PrivateRoute
              element={<HospitalStatistic />}
              redirectTo="/"
              blockList={['ROLE_ANONYMOUS']}
            />
          ),
        },
      ],
    },
  ]);

export default useMainRouter;
