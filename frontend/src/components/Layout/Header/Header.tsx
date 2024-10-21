import { ReactComponent as Logo } from '@assets/logo/logo.svg';
import { useAppDispatch, useAppSelector } from '@lib/hooks/redux';
import { logout } from '@store/member.slice';
import { useMemo } from 'react';
import { FaUserCircle } from 'react-icons/fa';
import { Link, useLocation } from 'react-router-dom';

const Header = () => {
  const member = useAppSelector((state) => state.member).value;
  const dispatch = useAppDispatch();
  const { pathname } = useLocation();

  const categories = useMemo(
    () => [
      { title: '모니터링', url: '/' },
      { title: '사고 기록', url: `/statistic/hospital` },
    ],
    [member],
  );
  return (
    <div className="border-border fixed left-0 top-0 z-10 flex h-[62px] w-full items-center justify-between gap-8 border-b bg-white px-8">
      <Link to="/">
        <Logo />
      </Link>
      {member.isAuth && member.role === 'ROLE_ADMIN' && (
        <div className="hidden flex-grow justify-start gap-10 md:flex">
          {categories.map(({ title, url }) => (
            <Link
              key={title}
              className={`${pathname === url && 'border-b-4 border-primary'} flex h-[62px] items-center justify-center px-2 text-base font-bold`}
              to={url}
            >
              <p className="">{title}</p>
            </Link>
          ))}
        </div>
      )}
      {member.isAuth ? (
        <div className="flex gap-4">
          <Link to="/my" className="flex gap-1">
            <FaUserCircle className="h-6 w-6 text-primary" />
            {member.role === 'ROLE_ADMIN' && member.name}
          </Link>
          <div className="border-l border-comment" />
          <button
            onClick={() => dispatch(logout())}
            className="font-bold text-comment"
          >
            로그아웃
          </button>
        </div>
      ) : (
        <Link to="/login" className="font-bold text-primary">
          로그인
        </Link>
      )}
    </div>
  );
};

export default Header;
