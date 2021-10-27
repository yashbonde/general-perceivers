YoCo (Remote Execution)
=======================

Code: `github <https://github.com/yashbonde/general-perceivers/tree/master/remote>`_, will eventually
be merged into the master of ``gperc``.

This is a new kind of programming concept I have been experimenting with. The way code is written
for networking systems, is very shallow and does not have a class (metaphorically). It is way too
complicated to use and build really cool apps. Current code has the following problems:

1. There is a big lag in getting feedback of why something failed on networks. In order to see why
   my code did not work I need to go through a maze of tabs (yes Kubernetes, pointing at you),
   that then tells me why something did not work.
2. Often there is a simple typo that leads to the crash, it takes in 0.5 seconds to fix it, but
   it took me over 40 seconds to figure out why. This latency is very bad, and outright biggest
   reason for slowdown of process.
3. Someone smart told me "In order to process something, you must control it, and to control it,
   you must measure it". Lately I have been obsessed with golang's function handling which forces
   returning the error. If you are writing things that talk to each other then they must be written
   in such a way the error is clearly conveyed.
4. Businesses and individuals alike will in near future have large amounts of compute at their
   disposal. This amount will be much much larger than what we have today and that will mostly be
   satisfied by server farms. Network businesses (facebook, netflix, etc.) will continue becoming
   more and more common replacing already failing desktop app market.
5. There needs to be an easy way to manage such machines and softwares running this. Testing will
   become more important and we all know how easy it currently is.

I use Python the most and it is the simplest, so taking inspiration from python module structure, we
can expose the methods over the internet as well. That is the application can be turned into a server
with a command. The same code can be used as an application ('client mode') and can be testes
('local mode') and can be served.

When we write client server applications there is a mismatch between the server functionalities (API
endpoints) and their actual location (codefile location in repo). This asymmetry means that you always
need to write client code and server code in two different ways. There are reasons why this is approach
works eg. apps that logically need only few endpoints (Instagram?), where interaction with the server
is low. But then there are other where the interaction with server is so high that it requires very
fast, reliable way to use it locally and on cloud with any noticeable difference.

**This means the behaviour of the code written must be the same cross format.**

For this I have written a simple decorator in python that takes in the path, return types, defaults and
reads annotation from python spec. This new approach is language agnostic because it's core objective
is not to tell syntax but the idea that cross format performance must be pre-tested and part of the
next generation of systems languages.

Not everything that shines if gold. What is that one problem that can be made into a business. As far
as I go, I think this suits perfectly to the applications that require precision control in high
compute environments, like orchestration of systems.
