#include <iostream>

#include <QCoreApplication>
#include <QDebug>

#include "service/Service.h"


int main(int argn, char* argv[])
{
    qSetMessagePattern("%{time hh:mm::ss.zzz} [%{type}] %{category}: %{message}");

    QCoreApplication app(argn, argv);

    service::Service service;
    service.start();

    return app.exec();
}
