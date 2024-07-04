package com.netease;

import io.grpc.Channel;
import io.grpc.Grpc;
import io.grpc.InsecureChannelCredentials;
import io.grpc.ManagedChannel;
import io.grps.protos.GrpsProtos;
import io.grps.protos.GrpsServiceGrpc;


public class GrpsClient {
    private final GrpsServiceGrpc.GrpsServiceBlockingStub blockingStub;

    public GrpsClient(Channel channel) {
        blockingStub = GrpsServiceGrpc.newBlockingStub(channel);
    }

    public void predictWithStr(String input) {
        final GrpsProtos.GrpsMessage grpsMessage = blockingStub.predict(GrpsProtos.GrpsMessage.newBuilder().setStrData(input).build());
        System.out.println("Predict response: " + grpsMessage.toString() + ", decoded str_data: " + grpsMessage.getStrData());
    }

    public static void main(String[] args) throws Exception {
        if (args == null || args.length != 2) {
            System.out.println("java -classpath target/*:maven-lib/* com.netease.GrpsClient <server> <inp>");
            return ;
        }

        String defaultCharset = System.getProperty("file.encoding");
        System.out.println(defaultCharset);
        String target = args[0];
        String input = args[1];

        System.out.println("grpc target: " + target + ", input: " + input);

        ManagedChannel channel = Grpc.newChannelBuilder(target, InsecureChannelCredentials.create())
                .build();
        try {
            GrpsClient client = new GrpsClient(channel);
            client.predictWithStr(input);
        } finally {
            channel.shutdownNow().awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS);
        }
    }
}
