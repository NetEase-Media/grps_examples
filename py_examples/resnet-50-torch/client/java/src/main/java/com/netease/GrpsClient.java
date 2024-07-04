package com.netease;

import com.google.protobuf.ByteString;
import io.grpc.Channel;
import io.grpc.Grpc;
import io.grpc.InsecureChannelCredentials;
import io.grpc.ManagedChannel;
import io.grps.protos.GrpsProtos;
import io.grps.protos.GrpsServiceGrpc;

import java.io.File;
import java.nio.file.Files;


public class GrpsClient {
    private final GrpsServiceGrpc.GrpsServiceBlockingStub blockingStub;

    public GrpsClient(Channel channel) {
        blockingStub = GrpsServiceGrpc.newBlockingStub(channel);
    }

    public void predictWithBinData(ByteString input) {
        final GrpsProtos.GrpsMessage grpsMessage =
                blockingStub.predict(GrpsProtos.GrpsMessage.newBuilder()
                        .setBinData(input).build());
        System.out.println(grpsMessage.toString());
    }

    public static void main(String[] args) throws Exception {
        if (args == null || args.length != 2) {
            System.out.println("java -classpath target/*:maven-lib/* com.netease.GrpsClient <server> <img_path>");
            return ;
        }

        String target = args[0];
        String img_path = args[1];

        System.out.println("grpc target: " + target + ", img_path: " + img_path);

        // read image file
        File file = new File(img_path);
        byte[] bytes = Files.readAllBytes(file.toPath());

        ManagedChannel channel = Grpc.newChannelBuilder(target, InsecureChannelCredentials.create())
                .build();
        try {
            ByteString input = ByteString.copyFrom(bytes);
            GrpsClient client = new GrpsClient(channel);
            client.predictWithBinData(input);
        } finally {
            channel.shutdownNow().awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS);
        }
    }
}
